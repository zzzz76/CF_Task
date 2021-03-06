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
class MFGeneral(object):

    def __init__(self, alpha, reg_u, reg_w, number_LatentFactors=10, number_epochs=10,
                 columns=["uid", "iid", "rating"]):
        self.alpha = alpha  # 学习率
        self.reg_u = reg_u  # P矩阵正则
        self.reg_w = reg_w  # Q矩阵正则
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs  # 最大迭代次数
        self.columns = columns

    def fit(self, trainset, testset):
        self.trainset = pd.DataFrame(trainset)
        self.testset = pd.DataFrame(testset)

        self.users_ratings = trainset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = trainset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.globalMean = self.trainset[self.columns[2]].mean()

        self.U, self.W, self.rmse, self.mae = self.train()

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
        for i in range(self.number_epochs):
            print("==========  epoch %d ==========" % i)
            U, W = self.sgd(U, W) # 每一轮更新 都要 计算 cost
            cost = self.cost(U, W)
            print("Training cost: ", cost)
            costs.append(cost)

            test_results = self.test(U, W)
            rmse, mae = accuray(test_results, method="all")
            # costs.append(rmse)
            print("Testing rmse: ", rmse, "mae: ", mae)

            if rmse < last_rmse:
                last_rmse = rmse
                last_mae = mae
                last_count = 0
            elif last_count < 4:
                last_count += 1
            else:
                break

        curve(costs, "mf_general")
        return U, W, last_rmse, last_mae


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

    def sgd(self, U, W):
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
                err = np.float32(r_ui - np.dot(v_u, v_i))

                v_u += self.alpha * (err * v_i - self.reg_u * v_u)
                v_i += self.alpha * (err * v_u - self.reg_w * v_i)


                U[uid] = v_u
                W[iid] = v_i
            except:
                print("+++++++++++++++++++")
                print(U[uid])
                print(W[iid])
                print(np.float32(r_ui - np.dot(U[uid], W[iid])))
                print("+++++++++++++++++++")

        return U, W

    def cost(self, U, W):
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
            cost += pow(r_ui - np.dot(v_u, v_i), 2)

        for uid in self.users_ratings.index:
            cost += self.reg_w * pow(np.linalg.norm(U[uid]), 2)

        for iid in self.items_ratings.index:
            cost += self.reg_u * pow(np.linalg.norm(W[iid]), 2)

        return cost/2


    def test(self, U, W):
        """
        测试数据集
        :param U: 用户隐空间矩阵
        :param W: 服务隐空间矩阵
        :return: 逐个返回测试评分
        """
        for uid, iid, real_rating in self.testset.itertuples(index=False):
            try:
                if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
                    pred_rating = self.globalMean
                else:
                    pred_rating = np.dot(U[uid], W[iid])
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating


if __name__ == '__main__':
    # training = "../dataset1/30/training.csv"
    # testing = "../dataset1/30/testing.csv"

    for i in [4]:
        print("----- Training Density %d/20 -----" % i)
        training = "../dataset1/" + str(i * 5) + "/training.csv"
        testing = "../dataset1/"+ str(i * 5) +"/testing.csv"

        print("load trainset: " + training)
        print("load testset:" + testing)

        # load data
        dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
        trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))
        testset = pd.read_csv(testing, usecols=range(3), dtype=dict(dtype))

        # training process
        mfg = MFGeneral(0.003, 0.02, 0.02, 10, 70, ["userId", "webId", "rating"])
        mfg.fit(trainset, testset)
        print("Final rmse: ", mfg.rmse, "mae: ", mfg.mae)