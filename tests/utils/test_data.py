from torch.utils.data import DataLoader
from rivertext.utils import TweetStream


def test_dataset_element_type():
    stream = TweetStream("./tests/tweets/1e5tweets.txt")
    for tweet in stream:
        assert isinstance(tweet, str)


def test_batch_size():
    stream = TweetStream("./tests/tweets/1e5tweets.txt")
    dataloader = DataLoader(stream, batch_size=32)
    for tweets in dataloader:
        assert len(tweets) == 32


def test_preprocess_method():
    stream = TweetStream("./tests/tweets/1e5tweets.txt")
    assert stream.preprocess("hello\n") == "hello"


def test_iter_method():
    stream = TweetStream("./tests/tweets/1e5tweets.txt")
    assert list(stream.__iter__())[1] == "I WANT CHOCOLATE p"
