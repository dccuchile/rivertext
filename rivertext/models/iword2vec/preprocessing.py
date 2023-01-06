""""""
import abc
import math
import random
from typing import Callable, List, Tuple

import numpy as np
import torch
from nltk import word_tokenize
from river.utils import dict2numpy

from rivertext.models.iword2vec.unigram_table import UnigramTable
from rivertext.utils import Vocab


class Preprocessing:
    """_summary_"""

    def __init__(
        self,
        vocab_size: int = 1_000_000,
        unigram_table_size: int = 100_000_000,
        window_size: int = 5,
        alpha: float = 0.75,
        subsampling_threshold: float = 1e-3,
        neg_samples_sum: int = 10,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
    ):

        """_summary_

        Parameters
        ----------
        vocab_size : int, optional
            _description_, by default 1_000_000
        unigram_table_size : int, optional
            _description_, by default 100_000_000
        window_size : int, optional
            _description_, by default 5
        alpha : float, optional
            _description_, by default 0.75
        subsampling_threshold : float, optional
            _description_, by default 1e-3
        neg_samples_sum : int, optional
            _description_, by default 10
        tokenizer : Callable[[str], List[str]], optional
            _description_, by default word_tokenize
        """
        self.vocab_size = vocab_size
        self.vocab = Vocab(vocab_size)

        self.total_counts = 0

        self.alpha = alpha

        self.window_size = window_size

        self.subsampling_threshold = subsampling_threshold

        self.unigram_table_size = unigram_table_size
        self.unigram_table = UnigramTable(self.unigram_table_size)

        self.tokenizer = tokenizer

        self.neg_samples_sum = neg_samples_sum

        np.random.seed(0)

    def reduce_vocab(self) -> None:
        """_summary_"""
        self.vocab.counter = self.vocab.counter - 1
        for idx, count in list(self.vocab.counter.items()):
            if count == 0:
                self.vocab.delete(idx)
        self.total_counts = np.sum(dict2numpy(self.vocab.counter))

    def rebuild_unigram_table(self) -> None:
        self.unigram_table.build(self.vocab, self.alpha)

    def update_unigram_table(self, word: str) -> None:
        """_summary_

        Parameters
        ----------
        word : str
            _description_
        """
        word_idx = self.vocab.add(word)
        self.total_counts += 1
        F = np.power(self.vocab.counter[word_idx], self.alpha) - np.power(
            (self.vocab.counter[word_idx] - 1), self.alpha
        )
        self.unigram_table.update(word_idx, F)

        if self.vocab_size == self.vocab.size:

            self.reduce_vocab()
            self.rebuild_unigram_table()

    def subsample_prob(self, word: str, t: float = 1e-3):
        z = self.vocab.counter[self.vocab.word2idx[word]] / self.total_counts
        return (math.sqrt(z / t) + 1) * t / z

    @abc.abstractmethod
    def __call__(self, batch: List[str]):
        """_summary_

        Parameters
        ----------
        batch : List[str]
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError()


class PrepCbow(Preprocessing):
    def __init__(
        self,
        vocab_size: int = 1_000_000,
        unigram_table_size: int = 100_000_000,
        window_size: int = 5,
        alpha: float = 0.75,
        subsampling_threshold: float = 1e-3,
        neg_samples_sum: int = 10,
        tokenizer=word_tokenize,
    ):

        super().__init__(
            vocab_size,
            unigram_table_size,
            window_size,
            alpha,
            subsampling_threshold,
            neg_samples_sum,
            tokenizer,
        )

    def __call__(
        self, batch: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : List[str]
            _description_

        Returns
        -------
        _type_
            _description_
        """
        targets = []
        contexts = []
        neg_samples = []
        for tweet in batch:

            for target_idx, token in enumerate(tweet):
                self.update_unigram_table(token)

            tokens_subsample = []

            for w in tweet:
                if w in self.vocab.word2idx and random.random() < self.subsample_prob(
                    w
                ):
                    tokens_subsample.append(w)

            n = len(tokens_subsample)
            if n < (self.window_size * 2 + 1):
                continue

            for idx, target_word in enumerate(tokens_subsample):
                target_word_idx = self.vocab.word2idx[target_word]
                contexts_words = get_contexts(idx, self.window_size, tokens_subsample)
                contexts_words_idxs = list(
                    map(lambda word: self.vocab.word2idx[word], contexts_words)
                )
                contexts_words_idxs += [
                    0 for _ in range(self.neg_samples_sum - len(contexts_words_idxs))
                ]
                neg_sample = self.unigram_table.samples(self.neg_samples_sum)

                targets.append(target_word_idx)
                contexts.append(contexts_words_idxs)
                neg_samples.append(neg_sample)

        return (
            torch.LongTensor(targets),
            torch.LongTensor(contexts),
            torch.LongTensor(neg_samples),
            len(batch),
        )


class PrepSG(Preprocessing):
    """"""

    def __init__(
        self,
        vocab_size: int = 1_000_000,
        unigram_table_size: int = 100_000_000,
        window_size: int = 5,
        alpha: float = 0.75,
        subsampling_threshold: float = 1e-3,
        neg_samples_sum: int = 10,
        tokenizer=word_tokenize,
    ):
        """_summary_

        Parameters
        ----------
        vocab_size : int, optional
            _description_, by default 1_000_000
        unigram_table_size : int, optional
            _description_, by default 100_000_000
        window_size : int, optional
            _description_, by default 5
        alpha : float, optional
            _description_, by default 0.75
        subsampling_threshold : float, optional
            _description_, by default 1e-3
        neg_samples_sum : int, optional
            _description_, by default 10
        tokenizer : _type_, optional
            _description_, by default word_tokenize
        """

        super().__init__(
            vocab_size,
            unigram_table_size,
            window_size,
            alpha,
            subsampling_threshold,
            neg_samples_sum,
            tokenizer,
        )

    def __call__(
        self, batch: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        targets = []
        contexts = []
        neg_samples = []
        for tweet in batch:

            for target_idx, token in enumerate(tweet):
                self.update_unigram_table(token)

            tokens_subsample = []

            for w in tweet:
                if w in self.vocab.word2idx and random.random() < self.subsample_prob(
                    w
                ):
                    tokens_subsample.append(w)

            n = len(tokens_subsample)
            if n < (self.window_size * 2 + 1):
                continue
            for index, target_word in enumerate(tokens_subsample):

                contexts_words = get_contexts(index, self.window_size, tokens_subsample)
                neg_sample = [
                    self.unigram_table.samples(self.neg_samples_sum)
                    for _ in range(len(contexts_words))
                ]

                targets += [self.vocab.word2idx[target_word]] * len(contexts_words)
                contexts += list(
                    map(lambda word: self.vocab.word2idx[word], contexts_words)
                )
                neg_samples += neg_sample

        return (
            torch.LongTensor(targets),
            torch.LongTensor(contexts),
            torch.LongTensor(neg_samples),
            len(batch),
        )


def get_contexts(ind_word, w_size, tokens):
    slice_start = ind_word - w_size if (ind_word - w_size >= 0) else 0
    slice_end = (
        len(tokens) if (ind_word + w_size + 1 >= len(tokens)) else ind_word + w_size + 1
    )
    first_part = tokens[slice_start:ind_word]
    last_part = tokens[ind_word + 1 : slice_end]
    contexts = list(first_part + last_part)
    return contexts
