"""
Used to split the data set

@author zzzz76
"""
import math
import pandas as pd
import numpy as np


def data_split(ratings, frac=0.8):
    '''
    split each user's rating data proportionally
    :param ratings: rating dataset
    :param part: number of parts
    :return: the partition index lists
    '''
    _ratings = ratings.reset_index()
    local_ratings = _ratings.groupby([_ratings['userRg'], _ratings['webRg']]).agg([list])

    testset_index = []
    for index_list, uid_list, iid_list, rating_list in local_ratings.itertuples(index=False):
        # 可以在ceil 处进行概率随机
        pos = round(len(index_list) * frac)
        # 因为不可变类型不能被 shuffle 方法作用，所以需要强行转换为列表
        _index_list = list(index_list)
        np.random.shuffle(_index_list)
        testset_index += list(_index_list[pos:])

    testset = ratings.loc[testset_index].reset_index(drop=True)
    trainset = ratings.drop(testset_index).reset_index(drop=True)
    return trainset, testset


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
            print("local mean is none")
        mean_col.append([mean])

    mean_col = pd.DataFrame(mean_col, columns=['mean'])
    dataset = pd.concat([dataset, mean_col], axis=1)
    return  dataset


if __name__ == '__main__':
    data_path = "../dataset1/ratings.csv"
    training = "../dataset1/25/training.csv"
    testing = "../dataset1/25/testing.csv"

    # set the type of data field to load
    dtype = {"userId": np.int32, "webId": np.int32, "rating": np.float32, "userRg": np.str_, "webRg": np.str_}
    # load the first 5 columns: user id, web id, rating, user region, web region
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(5))

    print("=========== split start =============")
    trainset, testset = data_split(ratings, 0.25)
    global_mean = trainset['rating'].mean()
    local_mean = ratings['rating'].groupby([ratings['userRg'], ratings['webRg']]).mean()
    # local_sums = trainset['rating'].groupby([trainset['userRg'], trainset['webRg']]).sum()
    # local_counts = trainset['rating'].groupby([trainset['userRg'], trainset['webRg']]).count()
    trainset = reprocess(trainset, global_mean, local_mean)
    testset = reprocess(testset, global_mean, local_mean)
    print("=========== split end =============")
    trainset.to_csv(training, index=False)
    testset.to_csv(testing, index=False)
    print("trainset save to: ", training)
    print("testset save to: ", testing)
