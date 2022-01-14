"""
Used to preprocess the raw data

@author zzzz76
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def load_data(data_path):
    """
    load data from data file
    :param data_path: the data file
    :return: the data matrix
    """
    R = []
    with open(data_path) as df:
        for line in df:
            R.append(list(map(float, line.split())))
        R = np.array(R)
    return R


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


def transform(R, I_outlier):
    """
    transform the data format from matrix to array
    :param R: data matrix
    :param I_outlier: outlier matrix
    :return: data array
    """
    tar_data = []
    m, n = R.shape

    for i in range(m):
        for j in range(n):
            if R[i][j] >= 0 and I_outlier[i][j] == 0:
                tar_data.append([i, j, R[i][j]])

    tar_data = pd.DataFrame(tar_data, columns=['userId', 'webId', 'rating'])
    return tar_data

if __name__ == '__main__':
    data_path = "../dataset2/rtMatrix.txt"
    tar_file = "../dataset2/ratings.csv"
    outlier_frac = float(0.2)
    print("=========== preprocess start =============")
    R = load_data(data_path)
    I_outlier = get_outlier(R, outlier_frac)
    tar_data = transform(R, I_outlier)
    tar_data.to_csv(tar_file, index=False)
    print("=========== preprocess end =============")