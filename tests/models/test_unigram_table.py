import numpy as np
import pytest
from rivertext.utils import Vocab
from rivertext.models.iword2vec import UnigramTable


np.random.seed(0)


def test_max_size_type():
    with pytest.raises(TypeError) as error:
        UnigramTable("5")
        assert "max_size should be int, got str" in error


def test_max_size():
    with pytest.raises(ValueError) as error:
        UnigramTable(-1)
        assert "max_size should be greater than 0, got -1" in error


def test_build_table():
    ut = UnigramTable(max_size=5)
    vocab = Vocab(3)
    vocab.add_tokens(["how", "are", "you", "?"])
    ut.build(vocab, alpha=0.75)
    for index in ut.table:
        print(ut.table)
        assert isinstance(index, float)
    table = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
    comparison = table == ut.table
    assert comparison.all()


def test_sample():
    ut = UnigramTable(max_size=5)
    vocab = Vocab(3)
    vocab.add_tokens(["how", "are", "you", "?"])
    ut.build(vocab, alpha=0.75)
    index = ut.sample()
    assert isinstance(index, float)
    assert index in np.array([0.0, 0.0, 1.0, 2.0, 2.0])


def test_samples():
    ut = UnigramTable(max_size=5)
    vocab = Vocab(3)
    vocab.add_tokens(["how", "are", "you", "?"])
    ut.build(vocab, alpha=0.75)
    indexes = ut.samples(2)
    for index in indexes:
        assert isinstance(index, float)
    assert indexes[0] in np.array([0.0, 0.0, 1.0, 2.0, 2.0])
    assert indexes[1] in np.array([0.0, 0.0, 1.0, 2.0, 2.0])
