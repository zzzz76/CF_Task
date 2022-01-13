

import numpy as np


from warnings import simplefilter
simplefilter('error')

from numpy import seterr
seterr(all='raise')

try:
    np.array([1]) / 0
except:
    print("ok")

a = 1.83164086e+10


print(a)
