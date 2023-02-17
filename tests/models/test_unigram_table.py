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
        assert "max_size should be greater than 100000, got -1" in error


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


def test_update_not_full_table():
    sentences = [["my", "name", "is", "John", "Doe"], ["how", "are", "you", "?"]]
    ut = UnigramTable(max_size=6)
    vocab = Vocab(3)
    total_counts = 0
    alpha = 1.3
    for sentence in sentences:
        for word in sentence:
            word_idx = vocab.add(word)
            if word_idx is not None:
                total_counts += 1
                F = np.power(vocab.counter[word_idx], alpha) - np.power(
                    (vocab.counter[word_idx] - 1), alpha
                )
                ut.update(word_idx, F)
    for index in ut.table:
        assert isinstance(index, float)


def test_update_not_full_table2():
    sentences = [
        line.split(" ") for line in open("/data/giturra/datasets/1e5tweets.txt")
    ]
    ut = UnigramTable(100_000)
    vocab = Vocab(1_000)
    total_counts = 0
    alpha = 1.3
    for sentence in sentences:
        for word in sentence:
            word_idx = vocab.add(word)
            if word_idx is not None:
                total_counts += 1
                F = np.power(vocab.counter[word_idx], alpha) - np.power(
                    (vocab.counter[word_idx] - 1), alpha
                )
                ut.update(word_idx, F)
    ut.build(vocab, alpha)
    for index in ut.table:
        assert isinstance(index, float)


def test_update_full_table():
    sentences = [
        ["my", "name", "is", "John", "Doe"],
        ["how", "are", "you", "?"],
        ["how", "are", "you", "?"],
    ]
    ut = UnigramTable(max_size=6)
    vocab = Vocab(10)
    total_counts = 0
    alpha = 0.75
    for sentence in sentences:
        for word in sentence:
            word_idx = vocab.add(word)
            if word_idx is not None:
                total_counts += 1
                F = np.power(vocab.counter[word_idx], alpha) - np.power(
                    (vocab.counter[word_idx] - 1), alpha
                )
                ut.update(word_idx, F)
    for index in ut.table:
        assert isinstance(index, float)
