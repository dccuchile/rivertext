from rivertext.utils import TweetStream


def test_dataset_element_type():
    stream = TweetStream("./tests/tweets/1e5tweets.txt")
    for tweet in stream:
        assert isinstance(tweet, str)
