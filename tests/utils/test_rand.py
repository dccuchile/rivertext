import numpy as np
from rivertext.utils import round_number


np.random.seed(0)


def test_round_number():
    assert 2 == round_number(1.2)
    assert 1 == round_number(0.7)
