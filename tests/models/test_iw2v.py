import numpy as np
from torch.utils.data import DataLoader
from rivertext.utils import TweetStream
from rivertext.models import IWord2Vec


def test_sg_learn_many():
    ts = TweetStream("./tests/tweets/tweets.txt")
    dataloader = DataLoader(ts, batch_size=32)
    iw2v = IWord2Vec(vocab_size=100_000)
    for tweet in dataloader:
        iw2v.learn_many(tweet)
    embs = iw2v.vocab2dict()

    for word, emb in embs.items():
        assert isinstance(word, str)
        assert isinstance(emb, np.ndarray)


def test_cbow_learn_many():
    ts = TweetStream("./tests/tweets/tweets.txt")
    dataloader = DataLoader(ts, batch_size=32)
    iw2v = IWord2Vec(vocab_size=100_000, sg=0)
    for tweet in dataloader:
        iw2v.learn_many(tweet)
    embs = iw2v.vocab2dict()

    for word, emb in embs.items():
        assert isinstance(word, str)
        assert isinstance(emb, np.ndarray)
