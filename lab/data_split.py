"""
Used to split the data set

@author zzzz76
"""
import pandas as pd
import numpy as np


def data_split(ratings, x=0.8):
    '''
    split each user's rating data proportionally
    :param ratings: rating dataset
    :param part: number of parts
    :param random: split randomly
    :return: the partition index lists
    '''
    trainset_index = []
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"] == uid).dropna()
        # 对指定用户的评分数据进行划分时，数据段的步长
        # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
        index = list(user_rating_data.index)
        np.random.shuffle(index)
        _index = round(len(user_rating_data) * x)
        trainset_index += list(index[:_index])
    return trainset_index


def reprocess(dataset, global_mean, local_means):
    """
    reprocess data with region-region ratings
    :param dataset: data without mean ratings
    :param global_mean: global mean rating
    :param local_means: local mean ratings
    :return: the dataset with mean ratings
    """
    mean_col = []
    for uid, iid, rating, urg, irg in dataset.itertuples(index=False):
        mean = local_means.get((urg, irg))
        if mean is None:
            mean = global_mean
        mean_col.append([mean])

    mean_col = pd.DataFrame(mean_col, columns=['mean'])
    dataset = pd.concat([dataset, mean_col], axis=1)
    return  dataset


if __name__ == '__main__':
    data_path = "../dataset1/ratings.csv"
    training = "../dataset1/8/training.csv"
    testing = "../dataset1/8/testing.csv"

    # set the type of data field to load
    dtype = {"userId": np.int32, "webId": np.int32, "rating": np.float32}
    # load the first 5 columns: user id, web id, rating, user region, web region
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))

    print("=========== split start =============")
    trainset_index = data_split(ratings, 0.08)
    print("=========== split end =============")

    trainset = ratings.loc[trainset_index].reset_index(drop=True)
    testset = ratings.drop(trainset_index).reset_index(drop=True)

    trainset.to_csv(training, index=False)
    testset.to_csv(testing, index=False)
    print("save trainset: " + training)
    print("save testset:" + testing)