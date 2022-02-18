import pandas as pd
import numpy as np
from lab.utils import accuray


def test(testset):
    for uid, iid, real_rating, mean in testset.itertuples(index=False):
        yield uid, iid, real_rating, mean


def reprocess(dataset, item_means):
    _dataset = []
    for uid, iid, rating in dataset.itertuples(index=False):
        mean = item_means.get(iid)
        if not mean is None:
            _dataset.append([uid, iid, rating, mean])

    _dataset = pd.DataFrame(_dataset, columns=['userId', 'webId', 'rating', 'mean'])
    return _dataset


if __name__ == '__main__':
    for i in [1]:
        print("----- Training Density %d/20 -----" % i)
        training = "../dataset1/" + str(i * 5) + "/training.csv"
        testing = "../dataset1/"+ str(i * 5) +"/testing.csv"
        print("load trainset: " + training)
        print("load testset:" + testing)

        # load data
        dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
        trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))
        testset = pd.read_csv(testing, usecols=range(3), dtype=dict(dtype))

        # catch item mean
        item_means = trainset['rating'].groupby(trainset['webId']).mean()
        testset = reprocess(testset, item_means)

        test_results = test(testset)
        rmse, mae = accuray(test_results, method="all")
        print("Testing rmse: ", rmse, "mae: ", mae)