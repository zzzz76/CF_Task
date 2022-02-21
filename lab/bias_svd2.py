"""
LFM Model

@author zzzz76
"""

import pandas as pd
import numpy as np
from lab.utils import accuray, curve

from warnings import simplefilter
simplefilter('error')

from numpy import seterr
seterr(all='raise')

# 评分预测    1-5
class Bias_svd(object):

    def __init__(self, eta, alpha, reg_u, reg_w, reg_bu, reg_bi, number_LatentFactors=10, number_epochs=10,
                 columns=["uid", "iid", "rating"]):
        self.eta = eta
        self.alpha = alpha  # 学习率
        self.reg_u = reg_u  # P矩阵正则
        self.reg_w = reg_w  # Q矩阵正则
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs  # 最大迭代次数
        self.columns = columns

    def fit(self, trainset, testset):
        self.trainset = pd.DataFrame(trainset)
        self.testset = pd.DataFrame(testset)

        self.users_ratings = trainset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = trainset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.globalMean = self.trainset[self.columns[2]].mean()
        # self.globalMean = 0.9181392788887024


        self.U, self.W, self.bu, self.bi, self.rmse, self.mae = self.train()

    def train(self):
        """
        训练模型
        :return: 隐空间矩阵
        """
        # test 用来防止 过拟合 以及 超参数的调节
        # 快速停止策略，如果test 连续5次增加
        last_rmse = 10
        last_mae = 10
        last_count = 0
        costs = []
        U, W = self._init_matrix() # 模型初始化
        bu, bi = self._init_bias() # 偏置初始化
        for i in range(self.number_epochs):

            # print(decayed_beta)
            print("==========  epoch %d ==========" % i)
            U, W, bu, bi = self.sgd(U, W, bu, bi) # 每一轮更新 都要 计算 cost
            cost = self.cost(U, W, bu, bi)
            print("Training cost: ", cost)
            costs.append(cost)

            test_results = self.test(U, W, bu, bi)
            rmse, mae = accuray(test_results, method="all")
            print("Testing rmse: ", rmse, "mae: ", mae)

            if rmse < last_rmse:
                last_rmse = rmse
                last_mae = mae
                last_count = 0
            elif last_count < 4:
                last_count += 1
            else:
                break

        curve(costs, "bias_svd")
        return U, W, bu, bi, last_rmse, last_mae


    def _init_matrix(self):
        """
        模型初始化，设置0,1之间的随机数为隐空间矩阵的初始值
        :return: 隐空间矩阵
        """
        # User-LF
        U = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float64)
        ))
        # Item-LF
        W = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float64)
        ))
        return U, W


    def _init_bias(self):
        """
        模型初始化，将用户偏置和服务偏置设置为0
        :return: 用户偏置 服务偏置
        """
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        return bu, bi


    def sgd(self, U, W, bu, bi):
        """
        使用随机梯度下降，优化模型
        :param U: 用户隐空间矩阵
        :param W: 服务隐空间矩阵
        :return: 经过优化的隐空间矩阵
        """
        for uid, iid, r_ui in self.trainset.itertuples(index=False):

            try:
                # User-LF U
                ## Item-LF W
                v_u = U[uid]  # 用户向量
                v_i = W[iid]  # 物品向量
                err = np.float32(r_ui - np.dot(v_u, v_i) * (1-self.eta) - (self.globalMean + bu[uid] + bi[iid]) * self.eta)

                v_u += self.alpha * (err * v_i * (1-self.eta) - self.reg_u * v_u)
                v_i += self.alpha * (err * v_u * (1-self.eta) - self.reg_w * v_i)

                U[uid] = v_u
                W[iid] = v_i

                bu[uid] += self.alpha * (err * self.eta - self.reg_bu * bu[uid])
                bi[iid] += self.alpha * (err * self.eta - self.reg_bi * bi[iid])

            except:
                print("+++++++++++++++++++")
                print(U[uid])
                print(W[iid])
                print(np.float32(r_ui - np.dot(U[uid], W[iid])))
                print("+++++++++++++++++++")

        return U, W, bu, bi

    def cost(self, U, W, bu, bi):
        """
        计算损失值
        :param U: 用户隐空间矩阵
        :param W: 服务隐空间矩阵
        :return: 模型损失值
        """
        cost = 0
        for uid, iid, r_ui in self.trainset.itertuples(index=False):
            v_u = U[uid]  # 用户向量
            v_i = W[iid]  # 物品向量
            cost += pow(r_ui - np.dot(v_u, v_i) * (1-self.eta) - (self.globalMean + bu[uid] + bi[iid]) * self.eta, 2)

        for uid in self.users_ratings.index:
            cost += self.reg_w * np.linalg.norm(U[uid]) + self.reg_bu * bu[uid]

        for iid in self.items_ratings.index:
            cost += self.reg_u * np.linalg.norm(W[iid]) + self.reg_bi * bi[iid]

        return cost


    def test(self, U, W, bu, bi):
        """
        测试数据集
        :param U: 用户隐空间矩阵
        :param W: 服务隐空间矩阵
        :return: 逐个返回测试评分
        """
        for uid, iid, real_rating in self.testset.itertuples(index=False):
            try:
                bias_u = 0
                bias_i = 0
                mf = 0
                if uid in self.users_ratings.index:
                    bias_u = bu[uid]
                if iid in self.items_ratings.index:
                    bias_i = bi[iid]
                if uid in self.users_ratings.index and iid in self.items_ratings.index:
                    mf = np.dot(U[uid], W[iid])
                pred_rating = mf * (1-self.eta) + (self.globalMean + bias_u + bias_i) * self.eta

            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating


def reprocess(dataset, local_means):
    """
    reprocess data with region-region ratings
    :param dataset: data without mean ratings
    :param local_means: local mean ratings
    :return: the dataset with mean ratings
    """
    _dataset = []
    for uid, iid, rating, urg, irg in dataset.itertuples(index=False):
        mean = local_means.get((urg, irg))
        if not mean is None:
            _dataset.append([uid, iid, rating])

    _dataset = pd.DataFrame(_dataset, columns=['userId','webId','rating'])
    return  _dataset


if __name__ == '__main__':
    # training = "../dataset1/30/training.csv"
    # testing = "../dataset1/30/testing.csv"

    for i in [2,3,4,5,6]:
        print("----- Training Density %d/20 -----" % i)
        training = "../dataset1/" + str(i * 5) + "/training.csv"
        testing = "../dataset1/"+ str(i * 5) +"/testing.csv"

        print("load trainset: " + training)
        print("load testset:" + testing)

        # load data
        dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32), ("userRg", np.str_), ("webRg", np.str_)]
        trainset = pd.read_csv(training, usecols=range(0,5), dtype=dict(dtype))
        testset = pd.read_csv(testing, usecols=range(0,5), dtype=dict(dtype))

        # catch local matrix
        local_means = trainset['rating'].groupby([trainset['userRg'], trainset['webRg']]).mean()

        trainset = reprocess(trainset, local_means)
        testset = reprocess(testset, local_means)

        # training process
        bsv = Bias_svd(0.5, 0.003, 0.02, 0.02, 0.02, 0.02, 10, 70, ["userId", "webId", "rating"])
        bsv.fit(trainset, testset)

        print("Final rmse: ", bsv.rmse, "mae: ", bsv.mae)