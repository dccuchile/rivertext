import pytest
from rivertext.utils import Vocab, Context


def test_max_size_type():
    with pytest.raises(TypeError) as error:
        vocab = Vocab("5")
        assert "max_size should be int, got -1" in error


def test_max_size():
    with pytest.raises(ValueError) as error:
        vocab = Vocab(-1)
        assert "max_size should be greater than 0, got -1" in error


def test_add_method():
    vocab = Vocab(max_size=3)
    assert len(vocab) == 0
    vocab.add("hello")
    assert len(vocab) == 1


def test_add_tokens_method():
    vocab = Vocab(max_size=3)
    vocab.add_tokens(["how", "are", "you", "?"])
    assert len(vocab) == 3


def test_is_full():
    vocab = Vocab(max_size=3)
    assert vocab.is_full() is False
    vocab.add_tokens(["how", "are", "you", "?"])
    assert vocab.is_full() is True


def test_contain():
    vocab = Vocab(max_size=3)
    vocab.add_tokens(["how", "are", "you", "?"])
    assert "hello" not in vocab
    assert "how" in vocab


def test_query_indexes():
    vocab = Vocab(max_size=3)
    assert vocab["hello"] == -1
    vocab.add("hello")
    assert vocab["hello"] == 0
    vocab.add("hello")
    assert vocab["hello"] == 0


def test_delete():
    vocab = Vocab(max_size=4)
    vocab.add_tokens(["how", "are", "you", "?"])
    assert len(vocab) == 4
    vocab.delete(0)
    assert len(vocab) == 3


def test_add_after_first_full():
    vocab = Vocab(max_size=4)
    vocab.add_tokens(["how", "are", "you", "?"])
    assert len(vocab) == 4
    vocab.delete(0)
    vocab.add("hi")
    assert len(vocab) == 4


def test_context_class():
    context = Context(3)
    context.add_tokens(["how", "are", "you", "?"])
    assert len(context) == 3
