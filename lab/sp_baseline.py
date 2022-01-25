from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic, NormalPredictor
from surprise import accuracy
from surprise.model_selection import KFold, split
from surprise import SVD,SVDpp

if __name__ == '__main__':

    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    data = Dataset.load_from_file('../datasets/ml-latest-small/ratings.csv', reader=reader)
    train_s,test_s = split.train_test_split(data, train_size=0.8)

algo1 = SVD()
    # algo2 = SVD(biased=False)
    # algo3 = SVDpp()

    print('SVDbias结果')
    algo1.fit(train_s)
    pre = algo1.test(test_s)
    accuracy.rmse(pre, verbose=True)
    print('SVD结果')
    # algo2.fit(train_s)
    # pre = algo2.test(test_s)
    # accuracy.rmse(pre, verbose=True)
    # print('SVD++结果')
    # algo3.fit(train_s)
    # pre = algo3.test(test_s)
    # accuracy.rmse(pre, verbose=True)


    # for i in [8]:
    #     print("----- Training Density %d/20 -----" % i)
    #     training = "../dataset1/" + str(i * 5) + "/training.csv"
    #     testing = "../dataset1/" + str(i * 5) + "/testing.csv"
    #
    #     print("load trainset: " + training)
    #     print("load testset:" + testing)
    #
    #     # load data
    #     dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32), ("mean", np.float32)]
    #     trainset = pd.read_csv(training, usecols=[0, 1, 2, 5], dtype=dict(dtype))
    #     testset = pd.read_csv(testing, usecols=[0, 1, 2, 5], dtype=dict(dtype))
    #
    #     # training process
    #     brm = Bias_rmf(0.004, 0.01, 0.01, 0.01, 0.01, 30, 300, ["userId", "webId", "rating", "mean"])
    #     brm.fit(trainset, testset)
    #
    #     print("Final rmse: ", brm.rmse, "mae: ", brm.mae)