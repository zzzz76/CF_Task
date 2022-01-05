"""
Used to preprocess the raw data

@author zzzz76
"""
import pandas as pd
import numpy as np

def transform(src_file, tar_file):
    """
    Transform the data format from matrix to array
    :param src_file: the source file
    :param tar_file: the target file
    """
    src_data = []
    tar_data = []
    with open(src_file) as sf:
        for line in sf:
            src_data.append(list(map(float, line.split())))
        src_data = np.array(src_data)

    m, n = src_data.shape
    for i in range(m):
        for j in range(n):
            rating = src_data[i][j]
            if rating > 0 and rating < 5:
                tar_data.append([i, j, rating])

    pd_data = pd.DataFrame(tar_data, columns=['userId', 'webId', 'rating'])
    pd_data.to_csv(tar_file, index=False)


if __name__ == '__main__':
    src_file = "dataset1/rtMatrix.txt"
    tar_file = "dataset1/ratings.csv"
    print("=========== transform start =============")
    transform(src_file, tar_file)
    print("=========== transform end =============")