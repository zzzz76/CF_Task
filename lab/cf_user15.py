import os

import pandas as pd
import numpy as np
from lab.utils import accuray, curve


def compute_pearson_similarity(ratings_matrix, based="user"):
    '''
    计算皮尔逊相关系数
    :param ratings_matrix: 用户-物品评分矩阵
    :param based: "user" or "item"
    :return: 相似度矩阵
    '''
    # 基于皮尔逊相关系数计算相似度
    # 用户相似度
    if based == "user":
        print("开始计算用户相似度矩阵")
        similarity = ratings_matrix.T.corr()
    elif based == "item":
        print("开始计算物品相似度矩阵")
        similarity = ratings_matrix.corr()
    else:
        raise Exception("Unhandled 'based' Value: %s"%based)
    print("相似度矩阵计算/加载完毕")
    return similarity

def predict(uid, iid, ratings_matrix, user_similar):
    '''
    预测给定用户对给定物品的评分值
    :param uid: 用户ID
    :param iid: 物品ID
    :param ratings_matrix: 用户-物品评分矩阵
    :param user_similar: 用户两两相似度矩阵
    :return: 预测的评分值
    '''
    # 1. 找出uid用户的相似用户
    similar_users = user_similar[uid].drop([uid]).dropna()
    # 相似用户筛选规则：正相关的用户
    similar_users = similar_users.where(similar_users>0).dropna()
    if similar_users.empty is True:
        raise Exception("用户<%d>没有相似的用户" % uid)

    # 2. 从uid用户的近邻相似用户中筛选出对iid物品有评分记录的近邻用户
    ids = set(ratings_matrix[iid].dropna().index)&set(similar_users.index)
    finally_similar_users = similar_users.loc[list(ids)]

    # 3. 结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    sum_up = 0    # 评分预测公式的分子部分的值
    sum_down = 0    # 评分预测公式的分母部分的值
    for sim_uid, similarity in finally_similar_users.iteritems():
        # 近邻用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # 近邻用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        # 计算分子的值
        sum_up += similarity * sim_user_rating_for_item
        # 计算分母的值
        sum_down += similarity

    # 计算预测的评分值并返回
    predict_rating = sum_up/sum_down
    return predict_rating

def test(testset, ratings_matrix, user_similar):
    count = 0
    for uid, iid, real_rating in testset.itertuples(index=False):
        try:
            pred_rating = predict(uid, iid, ratings_matrix, user_similar)
            count += 1

            if count % 100 == 1:
                print(count)
        except Exception as e:
            print("uid: ", uid, "iid: ", iid)
        else:
            yield uid, iid, real_rating, pred_rating


def filter_testset(testset):
    testset_index = []
    for uid in testset.groupby("userId").any().index:
        user_rating_data = testset.where(testset["userId"] == uid).dropna()
        # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
        index = list(user_rating_data.index)
        np.random.shuffle(index)  # 打乱列表
        _index = round(len(user_rating_data) * 0.1)
        testset_index += list(index[:_index])
    testset = testset.loc[testset_index].reset_index(drop=True)
    return testset


if __name__ == '__main__':
    training = "../dataset1/15/training.csv"
    testing = "../dataset1/95/testing.csv"

    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
    trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))
    testset = pd.read_csv(testing, usecols=range(3), dtype=dict(dtype))

    ratings_matrix = trainset.pivot_table(index=["userId"], columns=["webId"], values="rating")
    user_similar = compute_pearson_similarity(ratings_matrix, based="user")

    # testset = filter_testset(testset)
    print(len(testset))
    test_results = test(testset, ratings_matrix, user_similar)
    rmse, mae = accuray(test_results, method="all")
    print("Testing rmse: ", rmse, "mae: ", mae)

