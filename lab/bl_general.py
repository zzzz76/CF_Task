"""
Baseline Model

@author zzzz76
"""

import pandas as pd
import numpy as np
from lab.utils import accuray, curve

class BaselineCFBySGD(object):

    def __init__(self, number_epochs, alpha, reg, columns=["uid", "iid", "rating"]):
        # 梯度下降最高迭代次数
        self.number_epochs = number_epochs
        # 学习率
        self.alpha = alpha
        # 正则参数
        self.reg = reg
        # 数据集中user-item-rating字段的名称
        self.columns = columns

    def fit(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset
        # 用户评分数据
        self.users_ratings = trainset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = trainset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.trainset[self.columns[2]].mean()
        # 调用sgd方法训练模型参数
        self.bu, self.bi = self.train()

    def train(self):
        """
        训练模型
        :return:
        """
        tr_min = 10
        tr_last = 0
        costs = []
        bu, bi = self._init_bias()
        for i in range(self.number_epochs):
            print("==========  epoch %d ==========" % i)
            bu, bi = self.sgd(bu, bi)
            cost = self.cost(bu, bi)
            print("Training cost: ", cost)
            costs.append(cost)

            test_results = self.test(bu, bi)
            rmse, mae = accuray(test_results)
            print("Testing rmse: ", rmse, "mae: ", mae)

            if rmse < tr_min:
                tr_min = rmse
                tr_last = 0
            elif tr_last < 4:
                tr_last += 1
            else:
                break

        curve(costs)
        return bu, bi

    def _init_bias(self):
        """
        模型初始化，将用户偏置和服务偏置设置为0
        :return: 用户偏置 服务偏置
        """
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        return bu, bi

    def sgd(self, bu, bi):
        """
        利用随机梯度下降，优化bu，bi的值
        :param bu: 用户偏置向量
        :param bi: 服务偏置向量
        :return: 经过优化的偏置向量
        """
        for uid, iid, real_rating in self.trainset.itertuples(index=False):
            error = real_rating - (self.global_mean + bu[uid] + bi[iid])

            bu[uid] += self.alpha * (error - self.reg * bu[uid])
            bi[iid] += self.alpha * (error - self.reg * bi[iid])

        return bu, bi

    def cost(self, bu, bi):
        """
        计算损失值
        :param bu: 用户偏置向量
        :param bi: 服务偏置向量
        :return: 模型损失值
        """
        cost = 0
        for uid, iid, real_rating in self.trainset.itertuples(index=False):
            cost += pow(real_rating - (self.global_mean + bu[uid] + bi[iid]), 2)

        for uid in self.users_ratings.index:
            cost += self.reg * bu[uid]

        for iid in self.items_ratings.index:
            cost += self.reg * bi[iid]

        return cost


    def test(self, bu, bi):
        """
        测试数据集
        :param bu: 用户偏置向量
        :param bi: 服务偏置向量
        :return: 返回测试评分
        """
        for uid, iid, real_rating in self.testset.itertuples(index=False):
            try:
                if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
                    pred_rating = self.global_mean
                else:
                    pred_rating = self.global_mean + bu[uid] + bi[iid]
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

if __name__ == '__main__':
    training = "../dataset1/10/training.csv"
    testing = "../dataset1/10/testing.csv"

    # load data
    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
    trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))
    testset = pd.read_csv(testing, usecols=range(3), dtype=dict(dtype))

    # training process
    bcf = BaselineCFBySGD(100, 0.01, 0.01, ["userId", "webId", "rating"])
    bcf.fit(trainset, testset)
