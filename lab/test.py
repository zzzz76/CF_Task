

import numpy as np
import pandas as pd
import lab.utils as utils


from warnings import simplefilter
simplefilter('error')

from numpy import seterr
seterr(all='raise')

# try:
#     np.array([1]) / 0
# except:
#     print("ok")
#
# a = 1.83164086e+10
#
#
# print(a)


# beta = 0.01
# decay_rate = 0.9
# decay_steps = 5
# for i in range(300):
#     decayed_beta = beta * decay_rate**(i / decay_steps) + 0.01
#     print(decayed_beta)


# l = [i for i in range(15)]
# n = 3  #大列表中几个数据组成一个小列表
# print([l[i:i + n] for i in range(0, len(l), n)])


# names=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# n=3     #将列表每3个组成一个小列表，
# for i in range(0, len(names), n):
#     name=names[i:i + n]
#     print(name)
#


# index = list(names)
#
# print(index[120:])

# np.random.shuffle(index)
#
# for i in range(0, len(index), n):
#     name=names[i:i + n]
#     print(name)


# a = list([1,2,3])
# b = list([3,4,5])
# c = list([6,7,8])
#
#
# arr = [[] for i in range(3)]
# arr[0] += a
# arr[1] += b
# arr[2] += c
#
# print(arr)
# print(arr[1])
#
# arr[2] += c
# print(arr[2])
# arr[0] += c
# print(arr[0])
# arr[1] += c
# print(arr[1])
#
# arr[1][2] = -1
#
# print(arr[1])


# y = round(19/10)
# print(y)

# x = 10
# for i in range(1, x):
#     training = "../dataset1/" + str(x - i) + "/training.csv"
#     print(training)

# x = [1, 2, 3]
#
# y = [4, 5, 6]
#
# z = [7, 8, 9]
#
# xyz = dict(zip([1,2,4], [x,y,z]))
#
# print(xyz[4])


# list_data = [5,3,2]
# list_index = [('United states', 'Japan'), ('United states', 'United states'), ('Japan', 'Japan')]
#
# data = pd.Series(list_data, index= list_index)
#
# print(data.index)
# print(data)
# print(('United states', 'Japan') in data)

# user_path = "../dataset1/userlist.txt"
# web_path = "../dataset1/wslist.txt"
# ur_map = utils.load_map(user_path, 0, 2)
# wr_map = utils.load_map(web_path, 0, 4)
#
#
# print(4702 in ur_map.keys())
# print(4702 in wr_map.keys())
#
#
# print(4702 in ur_map)
# print(4702 in wr_map)

# import numpy as np
#
# # Save
# dictionary = {'hello':'world'}
# np.save('my_file.npy', dictionary)
#
# # Load
# read_dictionary = np.load('my_file.npy', allow_pickle=True).item()
# print(read_dictionary['hello']) # displays "world"

index = ('hello', 'world')
print(index[0])
print(index[1])

testset_index = []
index_list = [89,89,89,89,89,97,97,97,12,12,12]
pos = round(len(index_list) * 0.8)
_index_list = list(index_list)
np.random.shuffle(_index_list)
testset_index += list(_index_list[pos:])