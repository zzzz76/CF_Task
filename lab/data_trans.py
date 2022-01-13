"""
Used to preprocess the raw data

@author zzzz76
"""
import pandas as pd
import numpy as np

def transform(data_path):
    """
    Transform the data format from matrix to array
    :param data_path: the data path
    :return the target data
    """
    src_data = []
    tar_data = []
    with open(data_path) as df:
        for line in df:
            src_data.append(list(map(float, line.split())))
        src_data = np.array(src_data)

    m, n = src_data.shape
    for i in range(m):
        for j in range(n):
            rating = src_data[i][j]
            if rating > 0 and rating < 12:
                tar_data.append([i, j, rating])

    tar_data = pd.DataFrame(tar_data, columns=['userId', 'webId', 'rating'])
    return tar_data

if __name__ == '__main__':
    src_file = "../dataset1/rtMatrix.txt"
    tar_file = "../dataset1/ratings.csv"
    print("=========== transform start =============")
    tar_data = transform(src_file)
    tar_data.to_csv(tar_file, index=False)
    print("=========== transform end =============")