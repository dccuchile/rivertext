from math import ceil, floor

import numpy as np


def round_number(num):

    c = ceil(num)
    f = floor(num)
    uni = int(np.random.uniform(0.0, 1.0))
    if uni < (num - f):
        return c
    else:
        return f
