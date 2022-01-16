import numpy as np
import pandas as pd

def load_data(data_path, key, value):
    """
    load data map from data file
    :param data_path: data file
    :param key: column of key
    :param value: column of value
    :return: data map
    """
    id_list = []
    reg_list = []
    with open(data_path) as df:
        for line in df.readlines()[2:]:
            id_list.append(int(line.split('\t')[key]))
            reg_list.append(line.split('\t')[value])

        map = dict(zip(id_list, reg_list))
    return map

if __name__ == '__main__':
    # 加载 用户-区域字典 服务-区域字典
    user_path = "../dataset2/userlist.txt"
    web_path = "../dataset2/wslist.txt"

    ur_map = load_data(user_path, 0, 2)
    wr_map = load_data(web_path, 0, 4)

    # 加载 用户-服务 评分表
    training = "../dataset2/training.csv"
    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
    trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))

    # 按行遍历 评分表
    reg_table = []
    for uid, iid, real_rating in trainset.itertuples(index=False):
        reg_table.append([ur_map[uid], wr_map[iid], real_rating])
    reg_table = pd.DataFrame(reg_table, columns=['userReg', 'webReg', 'rating'])

    # 构建 地域 评分记录
    reg_table = reg_table['rating'].groupby([reg_table['userReg'], reg_table['webReg']])
    means = reg_table.mean()
    reg_data = means.unstack()

    print("ok")