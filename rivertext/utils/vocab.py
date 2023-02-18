"""Container class for saving vocabulary words"""
from typing import List, Set

from river.utils import VectorDict


class Vocab:
    """The Container class efficiently stores the mapping of words to their
    corresponding vector representations and supports all necessary elements
    required by an architecture, such as lookup tables, counters, and space indexes.
    It is an essential tool for managing and accessing word embeddings.

    References:
        1. https://github.com/yahoojapan/yskip/blob/master/src/vocab.h
    """

    def __init__(self, max_size: int = 1_000_000):
        """Initialize a Vocab instance

        Args:
            max_size:
                The size of the Vocabulary, by default 1_000_000.

        Raises:
            TypeError: The max size should be int number.
            ValueError: The max size should be greater than 0.
        """

        if not isinstance(max_size, int):
            raise TypeError(f"max_size should be int, got {max_size}")
        if max_size < 0:
            raise ValueError(f"max_size should be greater than 0, got {max_size}")

        self.max_size = max_size
        self.size = 0

        self.word2idx = VectorDict()
        self.idx2word = VectorDict()

        self.free_idxs: Set[int] = set()

        self.counter = VectorDict()

        self.first_full = False

    def add(self, word: str) -> int:
        """Add a new word.

        When a new word is added to the Vocabulary class, multiple data structures are
        updated to maintain the accuracy of the lookup tables, word counts, and
        available index for new entries.

        Args:
            word: New word to add.

        Returns:
            Index mapped to the new word. If the max size is equal to the current size,
                the method returns -1.
        """
        if word not in self.word2idx.keys() and not self.is_full():
            if not self.first_full:
                word_idx = self.size
            else:
                word_idx = self.free_idxs.pop()
            self.word2idx[word] = word_idx
            self.idx2word[word_idx] = word
            self.counter[word_idx] = 1
            self.size += 1

            if self.is_full():
                self.first_full = True
            return word_idx

        elif word in self.word2idx.keys():
            word_idx = self.word2idx[word]
            self.counter[word_idx] += 1
            return word_idx

    def add_tokens(self, tokens: List[str]) -> None:
        """Add a list of new words.

        Args:
            tokens: List of words to add.
        """
        for token in tokens:
            self.add(token)

    def is_full(self) -> bool:
        """Check if the vocabulary is full.

        Returns:
            True if the vocabulary structure is full, otherwise False.
        """
        return self.size == self.max_size

    def __len__(self) -> int:
        """Obtain the number of words inside the vocabulary.

        Returns:
            Number of words inside the vocabulary.
        """
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        """Check if a word is in the vocabulary.

        Args:
            word: Word to check.

        Returns:
            True if the word is in the vocabulary structure, otherwise False.
        """
        return word in self.word2idx

    def __getitem__(self, word: str) -> int:
        """Obtain the index of a given the word. If the word is not in the vocabulary
        returns -1.

        Args:
            word:
                word to get the index value.

        Returns:
            The value of index if the word is in the vocabulary, otherwise -1.
        """
        if word in self.word2idx:
            word_idx = self.word2idx[word]
            return word_idx
        return -1

    def delete(self, idx: int) -> None:
        """Delete the word mapped to the index idx.

        Args:
            idx:
                Index of the word.
        """
        self.free_idxs.add(idx)
        word = self.idx2word[idx]
        del self.word2idx[word]
        del self.idx2word[idx]
        del self.counter[idx]
        self.size -= 1


class Context(Vocab):
    """Container class for saving the contexts in the WCM model."""

    def __init__(self, max_size):
        super().__init__(max_size)
