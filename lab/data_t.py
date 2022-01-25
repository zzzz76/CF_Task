import numpy as np
import pandas as pd


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
    # 加载 用户-区域字典 服务-区域字典
    # 按照比例切割， 切割后计算平均值
    # 加载 用户-服务 评分表 dictionary
    training = "../dataset1/ratings.csv"
    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32), ("userRg", np.str_), ("webRg", np.str_)]
    trainset = pd.read_csv(training, usecols=range(5), dtype=dict(dtype))
    _trainset = trainset.reset_index()
    # users_ratings = trainset.groupby([trainset['userRg'], trainset['webRg']]).agg([list])[[self.columns[1], self.columns[2]]]
    local_ratings = _trainset.groupby([_trainset['userRg'], _trainset['webRg']]).agg([list])

    # 遍历其中的每一行
    testset_index = []
    for index_list, uid_list, iid_list, rating_list in local_ratings.itertuples(index=False):
        pos = round(len(index_list) * 0.8)
        # 因为不可变类型不能被 shuffle 方法作用，所以需要强行转换为列表
        _index_list = list(index_list)
        np.random.shuffle(_index_list)
        testset_index += list(_index_list[pos:])

    # trainset = trainset.loc[[1,3,4,2,6,8]]
    # trainset = trainset.reset_index(drop=True)
    #
    # 获取全局平均评分、局部平均评分
    # global_mean = trainset['rating'].mean()
    # local_means = trainset['rating'].groupby([trainset['userRg'], trainset['webRg']]).mean()
    # trainset = reprocess(trainset, global_mean, local_means)
    #
    print('ok')
