"""
Used to split the data set

@author zzzz76
"""
import pandas as pd
import numpy as np


def load_map(data_path, key, value):
    """
    load data map from data file
    :param data_path: data file
    :param key: column of key
    :param value: column of value
    :return: the data map
    """
    keys = []
    values = []
    with open(data_path) as df:
        for line in df.readlines()[2:]:
            keys.append(int(line.split('\t')[key]))
            values.append(line.split('\t')[value])

        map = dict(zip(keys, values))
    return map


def reprocess(dataset, ur_map, wr_map):
    _dataset = []
    for uid, iid, rating in dataset.itertuples(index=False):
        _dataset.append([uid, iid, rating, ur_map[uid], wr_map[iid]])

    _dataset = pd.DataFrame(_dataset, columns=['userId', 'webId', 'rating', 'userRg', 'webRg'])
    return _dataset


if __name__ == '__main__':
    user_path = "../dataset1/userlist.txt"
    web_path = "../dataset1/wslist.txt"
    # load user-region dict and web-region dict
    ur_map = load_map(user_path, 0, 2)
    wr_map = load_map(web_path, 0, 4)

    for i in [2,3,4,5,6]:
        print("----- Training Density %d/20 -----" % i)
        training = "../dataset1/" + str(i * 5) + "/training.csv"
        testing = "../dataset1/"+ str(i * 5) +"/testing.csv"
        print("load trainset: " + training)
        print("load testset:" + testing)

        # load data
        dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
        trainset = pd.read_csv(training, usecols=range(0,3), dtype=dict(dtype))
        testset = pd.read_csv(testing, usecols=range(0,3), dtype=dict(dtype))

        # reprocess
        trainset = reprocess(trainset, ur_map, wr_map)
        testset = reprocess(testset, ur_map, wr_map)

        trainset.to_csv(training, index=False)
        testset.to_csv(testing, index=False)
        print("save trainset: " + training)
        print("save testset:" + testing)
