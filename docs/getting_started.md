# Getting started

## Installation

RiverText is meant to work with Python 3.10 and above. Installation can be done via ```pip```:

```
pip install rivertext
```

Feel welcome to [open an issue on GitHub](https://github.com/dccuchile/rivertext/issues/new) if you are having any trouble.

## Usage

### Incremental Word Context Matrix

```python
>>> from rivertext.models.wcm import WordContextMatrix
>>> from torch.utils.data import DataLoader
>>> from rivertext.utils import TweetStream

>>> from web.datasets.similarity import fetch_MEN
>>> from web.evaluate import evaluate_similarity

>>> ts = TweetStream("/path/to/tweets.txt")

>>> wcm = WordContextMatrix(10000, 5, 500)

>>> dataloader = DataLoader(ts, batch_size=32)

>>> men = fetch_MEN()

>>> for batch in tqdm(dataloader):
...    wcm.learn_many(batch)

>>> embs = wcm.vocab2dict()
>>> result = evaluate_similarity(embs, men.X, men.y)
>>> print(f'Spearman Correlation: {result}')
Spearman Correlation: 0.08286971636085129
```

### Incremental Word2Vec
```python
>>> from torch.utils.data import DataLoader

>>> from rivertext.models.iw2v import IWord2Vec
>>> from rivertext.utils import TweetStream

>>> from web.datasets.similarity import fetch_MEN
>>> from web.evaluate import evaluate_similarity

>>> ts = TweetStream("/path/to/tweets.txt")
>>> men = fetch_MEN()
>>> dataloader = DataLoader(ts, batch_size=32)

>>> iw2v = IWord2Vec(window_size=3,
... emb_size=200,
... sg=1,
... neg_samples_sum=8,
... device="cuda:0"
... )

>>> for batch in tqdm(dataloader):
...    iw2v.learn_many(batch)

>>> embs = iw2v.vocab2dict()
>>> result = evaluate_similarity(embs, men.X, men.y)
>>> print(f'Spearman Correlation: {result}')
Spearman Correlation: 0.18149249269528014
```
