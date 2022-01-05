"""
Used to split the data set

@author zzzz76
"""
import pandas as pd
import numpy as np


def data_split(data_path, x=0.6, random=False):
    '''
    切分数据集， 这里为了保证用户数量保持不变，将每个用户的评分数据按比例进行拆分
    :param data_path: 数据集路径
    :param x: 训练集的比例，如x=0.8，则0.2是测试集
    :param random: 是否随机切分，默认False
    :return: 用户-物品评分矩阵
    '''

    # 设置要加载的数据字段的类型
    dtype = {"userId": np.int32, "webId": np.int32, "rating": np.float32}
    # 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))

    testset_index = []
    validset_index = []
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"] == uid).dropna()
        test_pos = round(len(user_rating_data) * x)
        valid_pos = round(len(user_rating_data) * (1 + x) / 2)

        if random:
            # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)  # 打乱列表
            testset_index += list(index[test_pos: valid_pos])
            validset_index += list(index[valid_pos:])
        else:
            # 将每个用户的x比例的数据作为训练集，剩余的作为测试集
            testset_index += list(user_rating_data.index.values[test_pos:valid_pos])
            validset_index += list(user_rating_data.index.values[valid_pos:])

    testset = ratings.loc[testset_index]
    validset = ratings.loc[validset_index]
    trainset = ratings.drop(testset_index).drop(validset_index)

    return trainset, testset, validset

if __name__ == '__main__':
    rating = "../dataset2/ratings.csv"
    training = "../dataset2/training.csv"
    testing = "../dataset2/testing.csv"
    validation = "../dataset2/validation.csv"

    print("=========== split start =============")
    trainset, testset, validset= data_split(rating, random=True)
    trainset.to_csv(training, index=False)
    testset.to_csv(testing, index=False)
    validset.to_csv(validation, index=False)
    print("=========== split end =============")
