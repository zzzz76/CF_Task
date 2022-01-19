"""
Used to preprocess the raw data

@author zzzz76
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def load_matrix(data_path):
    """
    load data matrix from data file
    :param data_path: data file
    :return: the data matrix
    """
    R = []
    with open(data_path) as df:
        for line in df:
            R.append(list(map(float, line.split())))
        R = np.array(R)
    return R


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


def preprocess(ur_map, wr_map, R, I_outlier):
    """
    preprocess the raw data
    :param ur_map: user-region map
    :param wr_map: web-region map
    :param R: user-web matrix
    :return: the target data
    """
    tar_data = []
    m, n = R.shape

    for i in range(m):
        for j in range(n):
            if R[i][j] >= 0 and j != 4700 and j != 4701 and I_outlier[i][j] != 1:
                tar_data.append([i, j, R[i][j], ur_map[i], wr_map[j]])

    tar_data = pd.DataFrame(tar_data, columns=['userId', 'webId', 'rating', 'userRg', 'webRg'])
    return tar_data

def get_outlier(R, outlier_frac):
    """
    get the outlier matrix
    :param R: data matrix
    :param outlier_frac: outlier fraction
    :return: outlier matrix
    """
    m,n = R.shape
    I_outlier = np.zeros([m,n])
    rng = np.random.RandomState(42)
    x = []
    x_ind = []
    for i in range(m):
        for j in range(n):
            if R[i][j] >=0:
                x.append(R[i][j])
                x_ind.append(i*n+j)

    x = np.array(x)
    x = x.reshape(-1,1)

    clf = IsolationForest(max_samples= len(x), random_state=rng, contamination=outlier_frac)
    clf.fit(x)
    y_pred_train = clf.predict(x)
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == -1:
            row = int(x_ind[i] / n)
            col = int(x_ind[i] % n)
            I_outlier[row][col] = 1

    return I_outlier


if __name__ == '__main__':
    user_path = "../dataset1/userlist.txt"
    web_path = "../dataset1/wslist.txt"
    data_path = "../dataset1/rtMatrix.txt"
    tar_file = "../dataset1/ratings.csv"

    # load user-region dict and web-region dict
    ur_map = load_map(user_path, 0, 2)
    wr_map = load_map(web_path, 0, 4)
    # load user-web matrix
    R = load_matrix(data_path)
    I_outlier = get_outlier(R, 0.02)
    print("=========== preprocess start =============")
    tar_data = preprocess(ur_map, wr_map, R, I_outlier)
    tar_data.to_csv(tar_file, index=False)
    print("=========== preprocess end =============")