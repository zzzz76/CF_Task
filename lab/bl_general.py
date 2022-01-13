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

    def fit(self, trainset, validset):
        self.trainset = trainset
        self.validset = validset
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
        vr_min = 10
        vr_last = 0
        costs = []
        bu, bi = self._init_bias()
        for i in range(self.number_epochs):
            print("==========  epoch %d ==========" % i)
            bu, bi = self.sgd(bu, bi)
            cost = self.cost(bu, bi)
            print("Training cost: ", cost)
            costs.append(cost)

            valid_results = self.valid(bu, bi)
            rmse = accuray(valid_results, method="rmse")
            print("Validation rmse: ", rmse)

            if rmse < vr_min:
                vr_min = rmse
                vr_last = 0
            elif vr_last < 5:
                vr_last += 1
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
            cost += self.reg * (pow(bu[uid], 2) * pow(bi[iid], 2))

        return cost / 2


    def valid(self, bu, bi):
        """
        验证数据集
        :param bu: 用户偏置向量
        :param bi: 服务偏置向量
        :return: 返回验证评分
        """
        for uid, iid, real_rating in self.validset.itertuples(index=False):
            try:
                pred_rating = self.global_mean + bu[uid] + bi[iid]
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

    def predict(self, uid, iid):
        '''评分预测'''
        if iid not in self.items_ratings.index:
            raise Exception("无法预测用户<{uid}>对电影<{iid}>的评分，因为训练集中缺失<{iid}>的数据".format(uid=uid, iid=iid))

        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating

    def test(self,testset):
        '''预测测试集数据'''
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
    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "webId", "rating"])
    bcf.fit(trainset, validset)

    # testing process
    pred_results = bcf.test(testset)
    rmse, mae = accuray(pred_results)

    print("rmse: ", rmse, "mae: ", mae)
