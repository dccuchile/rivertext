from math import ceil, floor

import numpy as np


def round_number(num: int) -> float:
    """Random rounder function.

    Parameters
    ----------
    num : int
        Number to round.

    Returns
    -------
    float
       Round number depeding on the uniform distribution,
       the number can be round by the ceil or floor.
    """
    c = ceil(num)
    f = floor(num)
    uni = int(np.random.uniform(0.0, 1.0))
    if uni < (num - f):
        return c
    else:
        return f
