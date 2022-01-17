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

    # 加载 用户-服务 评分表 dictionary
    training = "../dataset1/ratings.csv"
    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32), ("userRg", np.str_), ("webRg", np.str_)]
    trainset = pd.read_csv(training, usecols=range(5), dtype=dict(dtype))
    trainset = trainset.loc[[1,3,4,2,6,8]]
    trainset = trainset.reset_index(drop=True)

    # 获取全局平均评分、局部平均评分
    global_mean = trainset['rating'].mean()
    local_means = trainset['rating'].groupby([trainset['userRg'], trainset['webRg']]).mean()
    trainset = reprocess(trainset, global_mean, local_means)

    print('ok')
