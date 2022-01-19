"""
Used to split the data set

@author zzzz76
"""
import pandas as pd
import numpy as np


def data_split(ratings, part=10, random=False):
    '''
    split each user's rating data proportionally
    :param ratings: rating dataset
    :param part: number of parts
    :param random: split randomly
    :return: the partition index lists
    '''
    # 统计常见训练集占比的索引信息，第一行统计占比为1，第二行统计占比为9/10
    trainsets_index = [[] for i in range(part)]
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"] == uid).dropna()
        # 对指定用户的评分数据进行划分时，数据段的步长
        step = round(len(user_rating_data) / part)

        if random:
            # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)
            # 将每个用户(x-i)/x的数据作为训练集，剩余的作为测试集
            for i in range(part):
                train_pos = step * i
                trainsets_index[i] += list(index[train_pos:])
        else:
            # 将每个用户(x-i)/x的数据作为训练集，剩余的作为测试集
            for i in range(part):
                train_pos = step * i
                trainsets_index[i] += list(user_rating_data.index.values[train_pos:])

    return trainsets_index


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
    part = 20

    # set the type of data field to load
    dtype = {"userId": np.int32, "webId": np.int32, "rating": np.float32, "userRg": np.str_, "webRg": np.str_}
    # load the first 5 columns: user id, web id, rating, user region, web region
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(5))

    print("=========== split start =============")
    trainsets_index = data_split(ratings, part, random=True)
    print("=========== split end =============")

    for i in range(1, part):
        print("----- partition positon %d/%d -----" % (i, part))
        training = "../dataset1/" + str((part - i) * 5) + "/training.csv"
        testing = "../dataset1/" + str((part - i) * 5) + "/testing.csv"
        trainset = ratings.loc[trainsets_index[i]].reset_index(drop=True)
        testset = ratings.drop(trainsets_index[i]).reset_index(drop=True)

        # get the global mean and local means
        global_mean = trainset['rating'].mean()
        local_means = trainset['rating'].groupby([trainset['userRg'], trainset['webRg']]).mean()
        trainset = reprocess(trainset, global_mean, local_means)
        testset = reprocess(testset, global_mean, local_means)

        trainset.to_csv(training, index=False)
        testset.to_csv(testing, index=False)
        print("save trainset: " + training)
        print("save testset:" + testing)
