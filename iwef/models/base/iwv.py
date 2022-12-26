"""Contains a base class for implement any incremental method in IWEF."""
import abc
from typing import Callable, Dict, List, Tuple

import numpy as np
from river.base.transformer import Transformer
from river.feature_extraction.vectorize import VectorizerMixin


class IWVBase(Transformer, VectorizerMixin):
    """Base class for implement any incremental method in IWEF."""

    def __init__(
        self,
        vocab_size: int,
        vector_size: int,
        window_size: int,
        on: str = None,
        strip_accents: bool = True,
        lowercase: bool = True,
        preprocessor: Callable[str] = None,
        tokenizer: Callable[str, List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        """Base constructor for common hyperparameters.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.
        vector_size : int
            The dimension of the embedding.
        window_size : int
            The size of the window.
        on : str, optional
            The name of the feature that contains the text to vectorize. If `None`, then
            each `learn_one` and `transform_one` should treat `x` as a `str` and not as
            a `dict`., by default None.
        strip_accents : bool, optional
            Whether or not to strip accent characters, by default True.
        lowercase : bool, optional
            Whether or not to convert all characters to lowercase by default True.
        preprocessor : Callable[str], optional
            An optional preprocessing function which overrides the `strip_accents` and
            `lowercase` steps, while preserving the tokenizing and n-grams generation
            steps., by default None
        tokenizer : Callable[[str], List[str]], optional
            A function used to convert preprocessed text into a `dict` of tokens.
            A default tokenizer is used if `None` is passed. Set to `False` to disable
            tokenization, by default None.
        ngram_range : Tuple[int, int], optional
            The lower and upper boundary of the range n-grams to be extracted. All
            values of n such that `min_n <= n <= max_n` will be used. For example an
            `ngram_range` of `(1, 1)` means only unigrams, `(1, 2)` means unigrams and
            bigrams, and `(2, 2)` means only bigrams, by default (1, 1).
        """
        super().__init__(
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.vocab_size = vocab_size
        self.vector_size = vector_size
        self.window_size = window_size

    @abc.abstractmethod
    def learn_many(self, X: List[str], y=None, **kwargs) -> None:
        """Train a mini-batch of text features.

        Parameters
        ----------
        X : List[str]
            A list of sentence features.
        y : _type_, optional
            A series of target values, by default None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def vocab2dict(self) -> Dict[str, np.ndarray]:
        """
        Abstract method for transforming the vocabulary into a dictionary. The keys are
        the words of the vocabulary, and the values are the training vectors.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of embeddings.

        """
        raise NotImplementedError()
