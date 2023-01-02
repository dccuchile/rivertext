"""Incremental algorithm for extracting negative sampling from a text data stream."""
import numpy as np

from iwef.utils import Vocab, round_number


class UnigramTable:
    """The algorithm updates incrementally a unigram table, which Kaji
    and Kobayashi proposed.

    1. While the table is incomplete, it is updated as the original unigram table
    algorithm.

    2. If the table is complete, a random number n is selected, and n copies from the
    word w are added to the array table.

    References:
    | [1]: Nobuhiro Kaji and Hayato Kobayashi. 2017. Incremental Skip-gram Model
    |      with Negative Sampling. In Proceedings of the 2017 Conference on
    |      Empirical Methods in Natural Language Processing, pages 363–371,
    |      Copenhagen, Denmark. Association for Computational Linguistics.

    """

    def __init__(self, max_size: int = 100_000_000):
        """Initialize a Unigram Table instance.

        Args:
            max_size: Size of the unigram table, by default 100_000_000

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
        self.z = 0
        self.table = np.zeros(self.max_size)

    def sample(self) -> int:
        """Obtain a negative sample from the unigram table.

        Returns:
            Index of negative sample obtained.
        """
        assert 0 < self.size
        unigram_idx = self.table[np.random.randint(0, self.size)]
        return unigram_idx

    def samples(self, n: int) -> np.ndarray:
        """Obtain n negative samples from the unigram table

        Args:
            n: Number of negative samples.

        Returns:
            A array of negative samples.
        """
        unigram_idxs = list(self.table[np.random.randint(0, self.size, size=n)])
        return unigram_idxs

    def build(self, vocab: Vocab, alpha: float) -> None:
        """Build a unigram table based on the vocabulary structure.

        Args:
            vocab: Vocabulary.
            alpha: Smoothed parameter.
        """

        reserved_idxs = set(vocab.counter.keys())
        free_idxs = vocab.free_idxs
        counts = vocab.counter.to_numpy(reserved_idxs | free_idxs)
        vocab_size = len(counts)
        counts_pow = np.power(counts, alpha)
        z = np.sum(counts_pow)
        nums = self.max_size * counts_pow / z
        nums = np.vectorize(round_number)(nums)
        sum_nums = np.sum(nums)

        while self.max_size < sum_nums:
            w = int(np.random.randint(0, vocab_size))
            if 0 < nums[w]:
                nums[w] -= 1
                sum_nums -= 1

        self.z = z
        self.size = 0

        for w in range(vocab_size):
            self.table[self.size : self.size + nums[w]] = w
            self.size += nums[w]

    def update(self, word_idx: int, F: float) -> None:
        """Update the unigram table acording to the new words in the text stream.

        Args:
            word_idx: Index of the word to update in the unigram table.
            F: Normalize value.
        """

        assert 0 <= word_idx
        assert 0.0 <= F

        self.z += F
        if self.size < self.max_size:
            if F.is_integer():
                copies = min(int(F), self.max_size)
                self.table[self.size : self.size + copies] = word_idx
            else:
                copies = min(round_number(F), self.max_size)
                self.table[self.size : self.size + copies] = word_idx
            self.size += copies

        else:
            n = round_number((F / self.z) * self.max_size)
            for _ in range(n):
                table_idx = np.random.randint(0, self.max_size)
                self.table[table_idx] = word_idx
