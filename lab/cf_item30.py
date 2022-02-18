import pandas as pd
import numpy as np
from lab.utils import accuray

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

def predict(uid, iid, ratings_matrix, item_similar):
    '''
    预测给定用户对给定物品的评分值
    :param uid: 用户ID
    :param iid: 物品ID
    :param ratings_matrix: 用户-物品评分矩阵
    :param item_similar: 物品两两相似度矩阵
    :return: 预测的评分值
    '''
    # 1. 找出iid物品的相似物品
    similar_items = item_similar[iid].drop([iid]).dropna()
    # 相似物品筛选规则：正相关的物品
    similar_items = similar_items.where(similar_items>0).dropna()
    if similar_items.empty is True:
        raise Exception("物品<%d>没有相似的物品" %id)

    # 2. 从iid物品的近邻相似物品中筛选出uid用户评分过的物品
    ids = set(ratings_matrix.loc[uid].dropna().index)&set(similar_items.index)
    finally_similar_items = similar_items.loc[list(ids)]

    # 3. 结合iid物品与其相似物品的相似度和uid用户对其相似物品的评分，预测uid对iid的评分
    sum_up = 0    # 评分预测公式的分子部分的值
    sum_down = 0    # 评分预测公式的分母部分的值
    for sim_iid, similarity in finally_similar_items.iteritems():
        # 近邻物品的评分数据
        sim_item_rated_movies = ratings_matrix[sim_iid].dropna()
        # uid用户对相似物品物品的评分
        sim_item_rating_from_user = sim_item_rated_movies[uid]
        # 计算分子的值
        sum_up += similarity * sim_item_rating_from_user
        # 计算分母的值
        sum_down += similarity

    # 计算预测的评分值并返回
    predict_rating = sum_up/sum_down
    return predict_rating


def test(testset, ratings_matrix, item_similar):
    count = 0
    for uid, iid, real_rating in testset.itertuples(index=False):
        try:
            pred_rating = predict(uid, iid, ratings_matrix, item_similar)
            count += 1

            if count % 100 == 1:
                print(count)
        except Exception as e:
            print("uid: ", uid, "iid: ", iid)
        else:
            yield uid, iid, real_rating, pred_rating


if __name__ == '__main__':
    training = "../dataset1/30/training.csv"
    testing = "../dataset1/95/testing.csv"

    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
    trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))
    testset = pd.read_csv(testing, usecols=range(3), dtype=dict(dtype))

    ratings_matrix = trainset.pivot_table(index=["userId"], columns=["webId"], values="rating")
    item_similar = compute_pearson_similarity(ratings_matrix, based="item")

    print(len(testset))
    test_results = test(testset, ratings_matrix, item_similar)
    rmse, mae = accuray(test_results, method="all")
    print("Testing rmse: ", rmse, "mae: ", mae)