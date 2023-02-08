import numpy as np
from torch.utils.data import DataLoader
from rivertext.utils import TweetStream
from rivertext.models import WordContextMatrix


# def test_learn_one():
#     ts = TweetStream("./tests/tweets/1e5tweets.txt")
#     dataloader = DataLoader(ts, batch_size=1)
#     wcm = WordContextMatrix()
#     for tweet in dataloader:
#         wcm.learn_one(tweet)
#     embs = wcm.vocab2dict()

#     for word, emb in embs.items():
#         assert isinstance(word, str)
#         assert isinstance(emb, np.ndarray)


def test_learn_many():
    ts = TweetStream("./tests/tweets/1e5tweets.txt")
    dataloader = DataLoader(ts, batch_size=32)
    wcm = WordContextMatrix()
    for tweet in dataloader:
        wcm.learn_many(tweet)
    embs = wcm.vocab2dict()

    for word, emb in embs.items():
        assert isinstance(word, str)
        assert isinstance(emb, np.ndarray)


def test_learn_many_without_reduced_embs():
    ts = TweetStream("./tests/tweets/1e5tweets.txt")
    dataloader = DataLoader(ts, batch_size=32)
    wcm = WordContextMatrix(reduce_emb_dim=False)
    for tweet in dataloader:
        wcm.learn_many(tweet)
    embs = wcm.vocab2dict()

    for word, emb in embs.items():
        assert isinstance(word, str)
        assert isinstance(emb, np.ndarray)
