"""
Used to split the data set

@author zzzz76
"""
import pandas as pd
import numpy as np

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

    for i in [19]:
        print("----- Training Density %d/20 -----" % i)
        training = "../dataset1/" + str(i * 5) + "/training.csv"
        testing = "../dataset1/"+ str(i * 5) +"/testing.csv"

        print("load trainset: " + training)
        print("load testset:" + testing)

        # load data
        dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32), ("userRg", np.str_), ("webRg", np.str_)]
        trainset = pd.read_csv(training, usecols=range(0,5), dtype=dict(dtype))
        testset = pd.read_csv(testing, usecols=range(0,5), dtype=dict(dtype))

        # get the global mean and local means
        global_mean = trainset['rating'].mean()
        local_means = trainset['rating'].groupby([trainset['userRg'], trainset['webRg']]).mean()
        trainset = reprocess(trainset, global_mean, local_means)
        testset = reprocess(testset, global_mean, local_means)

        trainset.to_csv(training, index=False)
        testset.to_csv(testing, index=False)
        print("save trainset: " + training)
        print("save testset:" + testing)
