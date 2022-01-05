"""
Used to split the format data

@author zzzz76
"""
import pandas as pd
import numpy as np

def data_split(data_path, x=0.3, random=False):
    '''
    Split the datasets for training, validating and testing.
    The rating data of each user is split proportionally
    to keep the number of users constant.
    :param data_path: 数据集路径
    :param x: 训练集的比例，如x=0.8，则0.2是测试集
    :param random: 是否随机切分，默认False
    :return: 用户-物品评分矩阵
    '''
    print("开始切分数据集...")
    # 设置要加载的数据字段的类型
    dtype = {"userId": np.int32, "webId": np.int32, "rating": np.float32}
    # 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))

    testset_index = []
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"]==uid).dropna()
        if random:
            # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)    # 打乱列表
            _index = round(len(user_rating_data) * x)
            testset_index += list(index[_index:])
        else:
            # 将每个用户的x比例的数据作为训练集，剩余的作为测试集
            index = round(len(user_rating_data) * x)
            testset_index += list(user_rating_data.index.values[index:])

    testset = ratings.loc[testset_index]
    trainset = ratings.drop(testset_index)
    print("完成数据集切分...")
    return trainset, testset

if __name__ == '__main__':
    rating_file = "dataset1/ratings.csv"
    training_file = "dataset1/training.csv"
    testing_file = "dataset1/testing.csv"
    trainset, testset = data_split(rating_file, random=True)
    trainset.to_csv(training_file)
    testset.to_csv(testing_file)