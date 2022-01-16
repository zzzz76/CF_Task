import numpy as np
import pandas as pd
import lab.utils as utils

if __name__ == '__main__':
    # 加载 用户-区域字典 服务-区域字典
    user_path = "../dataset2/userlist.txt"
    web_path = "../dataset2/wslist.txt"

    ur_map = utils.load_map(user_path, 0, 2)
    wr_map = utils.load_map(web_path, 0, 4)

    # 加载 用户-服务 评分表 dictionary
    training = "../dataset2/training.csv"
    dtype = [("userId", np.int32), ("webId", np.int32), ("rating", np.float32)]
    trainset = pd.read_csv(training, usecols=range(3), dtype=dict(dtype))

    # 或者说在原始表中加入 数据





    # 构建 区域 评分记录
    regions_ratings = []
    for uid, iid, real_rating in trainset.itertuples(index=False):
        if uid in ur_map.keys() and iid in wr_map.keys():
            regions_ratings.append([ur_map[uid], wr_map[iid], real_rating])
    regions_ratings = pd.DataFrame(regions_ratings, columns=['userReg', 'webReg', 'rating'])
    regions_ratings = regions_ratings['rating'].groupby([regions_ratings['userReg'], regions_ratings['webReg']]).mean()


    # 在此处构建用户评分表 这里的每个数据都用实验一次
    # 直接在trainset 家一列 换句话 就是在此处进行拼接
    # 重新构造评分表 如果在原始表中有region就好的多

    # 获取 指定地域的评分


