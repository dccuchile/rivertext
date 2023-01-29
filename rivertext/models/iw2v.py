"""Implementation of the Incremental SkipGram and CBOW algorithms."""
from typing import Callable, Dict, List, Tuple

import numpy as np
from torch.optim import Optimizer, SparseAdam
from tqdm import tqdm

from rivertext.models.base import IWVBase
from rivertext.models.iword2vec import CBOW, SG, PrepCbow, PrepSG


class IWord2Vec(IWVBase):
    """Word2Vec incremental architectures is an adaptation of the popular word2vec
    proposed by Mikolov et al. to the streaming scenario. To adapt these algorithms to
    a streaming setting, we rely on the Incremental SkipGram with Negative Sampling
    model proposed by Kaji et al. The main assumptions we consider are:

    1. The models must deal with the fact that the vocabulary is dynamic and unknown,
        so the structures are updated as it is trained.
    2. The unigram table is created incrementally using the algorithm proposed by
        Kaji et al.
    3. The internal structure of the architecture was programmed in Pytorch.

    In this package, both CBOW and SG models were adapted using the incremental negative
        sampling technique to accelerate their training speed.

    References:
        1. Kaji, N., & Kobayashi, H. (2017). Incremental skip-gram model with negative
            sampling. arXiv preprint arXiv:1704.03956.
        2. Montiel, J., Halford, M., Mastelini, S. M., Bolmier, G., Sourty, R., Vaysse,
            R., ... & Bifet, A. (2021). River: machine learning for streaming data in
            Python.
    Examples:
        >>> from torch.utils.data import DataLoader
        >>> from rivertext.models.iw2v import IWord2Vec
        >>> from rivertext.utils import TweetStream
        >>> ts = TweetStream("/path/to/tweets.txt")
        >>> dataloader = DataLoader(ts, batch_size=32)
        >>> iw2v = IWord2Vec(
        ...    window_size=3,
        ...    vocab_size=3
        ...    emb_size=3,
        ...    sg=0,
        ...    neg_samples_sum=1,
        ...    device="cuda:0"
        ... )
        >>> for batch in dataloader:
        ...    iw2v.learn_many(batch)
        >>> iw2v.vocab2dict()
        {'hello': [0.77816248, 0.99913448, 0.14790398],
        'are': [0.86127345, 0.24901696, 0.28613529],
        'you': [0.64463917, 0.9003653 , 0.26000987],
        'this': [0.97007572, 0.08310498, 0.61532574],
        'example':  [0.74144294, 0.77877194, 0.67438642]
        }
        >>>  iw2v.transform_one('hello')
        [0.77816248, 0.99913448, 0.14790398]

    """

    def __init__(
        self,
        batch_size: int = 32,
        vocab_size: int = 1_000_000,
        emb_size: int = 100,
        unigram_table_size: int = 100_000_000,
        window_size: int = 5,
        alpha: float = 0.75,
        subsampling_threshold: float = 1e-3,
        neg_samples_sum: int = 10,
        sg: int = 1,
        lr: float = 0.025,
        device: str = None,
        optimizer: Optimizer = SparseAdam,
        on: str = None,
        strip_accents: bool = True,
        lowercase: bool = True,
        preprocessor=None,
        tokenizer: Callable[[str], List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        """An instance of IWord2Vec class.

        Args:
            batch_size: Mini-batch size, by default 32,
            vocab_size: Vocab size, by default 1_000_000.
            emb_size: Embdding size, by default 100.
            unigram_table_size: Unigram table size, by default 100_000_000.
            window_size: Window size, by default 5
            alpha: Smoother parameter, by default 0.75
            subsampling_threshold : Subsampling parameter, by default 1e-3
            neg_samples_sum: Number of negative sampling to used, by default 10.
            sg: training algorithm, 1 for CBOW; otherwise SG.
            lr: Learning rate of the optimizer, by default 0.025
            device: Device to run the wrapped model on. Can be "cpu" or "cuda", by
                default cuda.
            optimizer: Optimizer to be used for training the model.,
                by default SparseAdam.
            on: The name of the feature that contains the text to vectorize. If `None`,
                then each `learn_one` and `transform_one` should treat `x` as a `str`
                and not as a `dict`., by default None.
            strip_accents: Whether or not to strip accent characters, by default True.
                lowercase: Whether or not to convert all characters to lowercase
                by default True.
            preprocessor: An optional preprocessing function which overrides the
                `strip_accents` and `lowercase` steps, while preserving the tokenizing
                and n-grams generation steps., by default None
            tokenizer: A function used to convert preprocessed text into a `dict` of
                tokens. A default tokenizer is used if `None` is passed. Set to `False`
                to disable tokenization, by default None.
            ngram_range: The lower and upper boundary of the range n-grams to be
                extracted. All values of n such that `min_n <= n <= max_n` will be used.
                For example an `ngram_range` of `(1, 1)` means only unigrams, `(1, 2)`
                means unigrams and bigrams, and `(2, 2)` means only bigrams, by default
                (1, 1).

        """

        super().__init__(
            vocab_size,
            emb_size,
            window_size,
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.neg_sample_num = neg_samples_sum
        self.sg = sg

        if sg:
            self.model_name = "SG"
            self.model = SG(self.vocab_size, emb_size)
            self.prep = PrepSG(
                vocab_size=vocab_size,
                unigram_table_size=unigram_table_size,
                window_size=window_size,
                alpha=alpha,
                subsampling_threshold=subsampling_threshold,
                neg_samples_sum=neg_samples_sum,
                tokenizer=tokenizer,
            )
            self.optimizer = optimizer(self.model.parameters(), lr=lr)

        else:
            self.model_name = "CBOW"
            self.model = CBOW(vocab_size, emb_size)
            self.prep = PrepCbow(
                vocab_size=vocab_size,
                unigram_table_size=unigram_table_size,
                window_size=window_size,
                alpha=alpha,
                subsampling_threshold=subsampling_threshold,
                neg_samples_sum=neg_samples_sum,
                tokenizer=tokenizer,
            )
            self.optimizer = optimizer(self.model.parameters(), lr=0.05)
        self.device = device
        self.model.to(self.device)

    def vocab2dict(self) -> Dict[str, np.ndarray]:
        """Converts the vocabulary in a dictionary of embeddings.

        Returns:
            An dict where the words are the keys, and their values are the
                embedding vectors.
        """
        embeddings = {}
        for word in tqdm(self.prep.vocab.word2idx.keys()):
            embeddings[word] = self.transform_one(word)
        return embeddings

    def transform_one(self, x: str) -> np.ndarray:
        """Obtain the vector embedding of a word.

        Args:
            x: word to obtain the embedding.

        Returns:
            The vector embedding of the word.
        """
        word_idx = self.prep.vocab[x]
        return self.model.get_embedding(word_idx)

    def learn_one(self, x: str, **kwargs) -> None:
        """Train one instance of text feature.

        Args:
            x: one line of text.

        Examples:
            >>> from torch.utils.data import DataLoader
            >>> from rivertext.models.iw2v import IWord2Vec
            >>> from rivertext.utils import TweetStream
            >>> ts = TweetStream("/path/to/tweets.txt")
            >>> dataloader = DataLoader(ts)
            >>> iw2v = IWord2Vec(
            ...    window_size=3,
            ...    vocab_size=3
            ...    emb_size=3,
            ...    sg=0,
            ...    neg_samples_sum=1,
            ...    device="cuda:0"
            ... )
            >>> for tweet in dataloader:
            ...    iw2v.learn_one(tweet)
            >>> iw2v.vocab2dict()
            {'hello': [0.77816248, 0.99913448, 0.14790398],
            'are': [0.86127345, 0.24901696, 0.28613529],
            'you': [0.64463917, 0.9003653 , 0.26000987],
            'this': [0.97007572, 0.08310498, 0.61532574],
            'example':  [0.74144294, 0.77877194, 0.67438642]
            }
            >>>  iw2v.transform_one('hello')
            [0.77816248, 0.99913448, 0.14790398]

        """
        tokens = self.process_text(x)
        batch = self.prep(tokens)
        targets = batch[0].to(self.device)
        contexts = batch[1].to(self.device)
        neg_samples = batch[2].to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(targets, contexts, neg_samples)
        loss.backward()
        self.optimizer.step()

    def learn_many(self, X: List[str], y=None, **kwargs) -> None:
        """Train a mini-batch of text features.

        Args:
            X: A list of sentence features.
            y: A series of target values, by default None.

        Examples:
            >>> from torch.utils.data import DataLoader
            >>> from rivertext.models.iw2v import IWord2Vec
            >>> from rivertext.utils import TweetStream
            >>> ts = TweetStream("/path/to//tweets.txt")
            >>> dataloader = DataLoader(ts, batch_size=32)
            >>> iw2v = IWord2Vec(
            ...    window_size=3,
            ...    vocab_size=3
            ...    emb_size=3,
            ...    sg=0,
            ...    neg_samples_sum=1,
            ...    device="cuda:0"
            ... )
            >>> for batch in dataloader:
            ...    iw2v.learn_many(batch)
            >>> iw2v.vocab2dict()
            {'hello': [0.77816248, 0.99913448, 0.14790398],
            'are': [0.86127345, 0.24901696, 0.28613529],
            'you': [0.64463917, 0.9003653 , 0.26000987],
            'this': [0.97007572, 0.08310498, 0.61532574],
            'example':  [0.74144294, 0.77877194, 0.67438642]
            }
            >>>  wcm.transform_one('hello')
            [0.77816248, 0.99913448, 0.14790398]
        """

        tokens = list(map(self.process_text, X))
        batch = self.prep(tokens)
        targets = batch[0].to(self.device)
        contexts = batch[1].to(self.device)
        neg_samples = batch[2].to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(targets, contexts, neg_samples)
        loss.backward()
        self.optimizer.step()
