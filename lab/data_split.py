"""
Used to split the data set

@author zzzz76
"""
import pandas as pd
import numpy as np


def data_split(ratings, x=10, random=False):
    '''
    切分数据集， 这里为了保证用户数量保持不变，将每个用户的评分数据按比例进行拆分
    :param ratings: 评分数据集
    :param x: 训练集的比例，如x=0.8，则0.2是测试集
    :param random: 是否随机切分，默认False
    :return: 用户-物品评分矩阵
    '''
    # 统计常见训练集占比的索引信息，第一行统计占比为1，第二行统计占比为9/10
    trainsets_index = [[] for i in range(x)]
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"] == uid).dropna()
        # 对指定用户的评分数据进行划分时，数据段的步长
        step = round(len(user_rating_data) / x)

        if random:
            # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)
            # 将每个用户(x-i)/x的数据作为训练集，剩余的作为测试集
            for i in range(x):
                train_pos = step * i
                trainsets_index[i] += list(index[train_pos:])
        else:
            # 将每个用户(x-i)/x的数据作为训练集，剩余的作为测试集
            for i in range(x):
                train_pos = step * i
                trainsets_index[i] += list(user_rating_data.index.values[train_pos:])
    return trainsets_index


if __name__ == '__main__':
    data_path = "../dataset1/ratings.csv"
    x = 10

    # 设置要加载的数据字段的类型
    dtype = {"userId": np.int32, "webId": np.int32, "rating": np.float32, "userRg": np.str_, "webRg": np.str_}
    # 这里我们加载前五列数据，分别为用户编号，服务编号，评分，用户区域，服务区域
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(5))

    print("=========== split start =============")
    trainsets_index = data_split(ratings, x, random=True)
    print("=========== split end =============")

    for i in range(1, x):
        print("----- partition positon %d/%d -----" % (i, x))
        training = "../dataset1/" + str(x - i) + "0/training.csv"
        testing = "../dataset1/" + str(x - i) + "0/testing.csv"
        trainset = ratings.loc[trainsets_index[i]]
        testset = ratings.drop(trainsets_index[i])
        trainset.to_csv(training, index=False)
        testset.to_csv(testing, index=False)
        print("save trainset: " + training)
        print("save testset:" + testing)
