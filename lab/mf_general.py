'''
LFM Model
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def accuray(predict_results, method="all"):
    '''
    准确性指标计算方法
    :param predict_results: 预测结果，类型为容器，每个元素是一个包含uid,iid,real_rating,pred_rating的序列
    :param method: 指标方法，类型为字符串，rmse或mae，否则返回两者rmse和mae
    :return:
    '''

    def rmse(predict_results):
        '''
        rmse评估指标
        :param predict_results:
        :return: rmse
        '''
        length = 0
        _rmse_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
        return round(np.sqrt(_rmse_sum / length), 4)

    def mae(predict_results):
        '''
        mae评估指标
        :param predict_results:
        :return: mae
        '''
        length = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _mae_sum += abs(pred_rating - real_rating)
        return round(_mae_sum / length, 4)

    def rmse_mae(predict_results):
        '''
        rmse和mae评估指标
        :param predict_results:
        :return: rmse, mae
        '''
        length = 0
        _rmse_sum = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
            _mae_sum += abs(pred_rating - real_rating)
        return round(np.sqrt(_rmse_sum / length), 4), round(_mae_sum / length, 4)

    if method.lower() == "rmse":
        return rmse(predict_results)
    elif method.lower() == "mae":
        return mae(predict_results)
    else:
        return rmse_mae(predict_results)

def curve(costs):
    n = len(costs)
    x = range(n)
    plt.plot(x, costs, color='r', linewidth=3)
    plt.title("Convergence curve")
    plt.xlabel("generation")
    plt.ylabel("loss")
    plt.show()

# 评分预测    1-5
class LFM(object):

    def __init__(self, alpha, reg_p, reg_q, number_LatentFactors=10, number_epochs=10,
                 columns=["uid", "iid", "rating"]):
        self.alpha = alpha  # 学习率
        self.reg_p = reg_p  # P矩阵正则
        self.reg_q = reg_q  # Q矩阵正则
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs  # 最大迭代次数
        self.columns = columns

    def fit(self, dataset, validset):
        self.dataset = pd.DataFrame(dataset)
        self.validset = pd.DataFrame(validset)

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q = self.train()

    def train(self):
        """
        训练模型
        :return: 隐空间矩阵
        """

        # valid 用来防止 过拟合 以及 超参数的调节
        # 快速停止策略，如果valid 连续5次增加
        vr_min = 10
        vr_last = 0
        costs = []
        P, Q = self._init_matrix()
        for i in range(self.number_epochs):
            print("==========  epoch %d ==========" % i)
            P, Q = self.sgd(P, Q)
            cost = self.cost(P, Q)
            print("Training cost: ", cost)
            costs.append(cost)
            if cost < 0.01:
                break

            valid_results = lfm.valid(P, Q)
            rmse = accuray(valid_results, method="rmse")
            print("Validation rmse: ", rmse)

            if rmse < vr_min:
                vr_min = rmse
                vr_last = 0
            elif vr_last < 4:
                vr_last += 1
            else:
                break

        curve(costs)
        return P, Q


    def _init_matrix(self):
        """
        模型初始化，设置0,1之间的随机数为隐空间矩阵的初始值
        :return: 隐空间矩阵
        """
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def sgd(self, P, Q):
        """
        使用随机梯度下降，优化模型
        :param P: 用户隐空间矩阵
        :param Q: 服务隐空间矩阵
        :return: 经过优化的隐空间矩阵
        """
        for uid, iid, r_ui in self.dataset.itertuples(index=False):
            # User-LF P
            ## Item-LF Q
            v_pu = P[uid]  # 用户向量
            v_qi = Q[iid]  # 物品向量
            err = np.float32(r_ui - np.dot(v_pu, v_qi))

            v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
            v_qi += self.alpha * (err * v_pu - self.reg_q * v_qi)

            P[uid] = v_pu
            Q[iid] = v_qi
        return P, Q

    def cost(self, P, Q):
        """
        计算损失值
        :param P: 用户隐空间矩阵
        :param Q: 服务隐空间矩阵
        :return: 模型损失值
        """
        cost = 0
        for uid, iid, r_ui in self.dataset.itertuples(index=False):
            v_pu = P[uid]  # 用户向量
            v_qi = Q[iid]  # 物品向量
            cost += pow(r_ui - np.dot(v_pu, v_qi), 2)

            for k in range(self.number_LatentFactors):
                cost += self.reg_p * pow(v_pu[k], 2) + self.reg_q * pow(v_qi[k], 2)

        return cost / 2

    def valid(self, P, Q):
        """
        验证数据集
        :param P: 用户隐空间矩阵
        :param Q: 服务隐空间矩阵
        :return: 逐个返回验证评分
        """
        for uid, iid, real_rating in self.validset.itertuples(index=False):
            try:
                pred_rating = np.dot(P[uid], Q[iid])
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

    def predict(self, uid, iid):
        """
        评分预测
        :param uid: 用户id
        :param iid: 服务id
        :return: 评分预测值
        """
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]
        q_i = self.Q[iid]

        return np.dot(p_u, q_i)

    def test(self, testset):
        """
        测试数据集
        :param testset: 测试集
        :return: 逐个返回预测评分
        """
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating


if __name__ == '__main__':
    training = "../dataset2/training.csv"
    testing = "../dataset2/testing.csv"
    validation = "../dataset2/validation.csv"

    # load data
    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
    trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))
    testset = pd.read_csv(testing, usecols=range(3), dtype=dict(dtype))
    validset = pd.read_csv(validation, usecols=range(3), dtype=dict(dtype))

    # training process
    lfm = LFM(0.02, 0.01, 0.01, 10, 300, ["userId", "webId", "rating"])
    lfm.fit(trainset, validset)

    # testing process
    pred_results = lfm.test(testset)
    rmse, mae = accuray(pred_results)

    print("rmse: ", rmse, "mae: ", mae)
