import pandas as pd
import numpy as np

src_data = []
tar_data = []

m = 10
n = 100
for i in range(m):
    for j in range(n):
        tar_data.append([i, j, 0.123])

# tar_data = np.array(tar_data)
# print(tar_data)
pd_data = pd.DataFrame(tar_data, columns=['userId', 'movieId', 'rating'])
print(pd_data)
