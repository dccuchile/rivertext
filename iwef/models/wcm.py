"""Implementation of the Incremental Word Contex Matrix algorithm."""
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import IncrementalPCA

from iwef.models.base import IWVBase
from iwef.utils import Context, Vocab


class WordContextMatrix(IWVBase):
    """

    Examples:
        >>> from iwef.models.wcm import WordContextMatrix
        >>> from torch.utils.data import DataLoader
        >>> from iwef.utils import TweetStream
        >>> ts = TweetStream("/path/to/tweets.txt")
        >>> wcm = WordContextMatrix(5, 1, 3)
        >>> dataloader = DataLoader(ts, batch_size=5)
        >>> for batch in tqdm(dataloader):
        >>>     wcm.learn_many(batch)
        >>> wcm.vocab2dict()
        {'hello': [0.77816248, 0.99913448, 0.14790398],
        'are': [0.86127345, 0.24901696, 0.28613529],
        'you': [0.64463917, 0.9003653 , 0.26000987],
        'this': [0.97007572, 0.08310498, 0.61532574],
        'example':  [0.74144294, 0.77877194, 0.67438642]
        }
    """

    def __init__(
        self,
        vocab_size: int,
        window_size: int,
        context_size: int,
        emb_size: int = 300,
        reduce_emb_dim: bool = True,
        on: str = None,
        strip_accents: bool = True,
        lowercase: bool = True,
        preprocessor: Callable = None,
        tokenizer: Callable[[str], List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        """An instance of WCM class.

        Args:
            vocab_size: The size of the vocabulary.
            window_size: The size of the window.
            context_size: The size of the contexts.
            emb_size: The size of the embeddings.
            reduce_emb_dim: , by default True
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

        self.context_size = context_size
        self.vocab = Vocab(self.vocab_size)
        self.contexts = Context(self.context_size)
        self.coocurence_matrix = sparse.lil_matrix((self.vocab_size, self.context_size))

        self.d = 0

        self.reduced_emb_dim = reduce_emb_dim
        if self.reduced_emb_dim:

            self.modified_words = set()
            self.emb_matrix = sparse.lil_matrix((self.vocab_size, self.vector_size))

            self.transformer = IncrementalPCA(
                n_components=self.vector_size, batch_size=300
            )

    def learn_one(self, x: str, **kwargs) -> None:
        """Train one instance using SPMMI method.

        Args:
            x: one line of text.
        """
        tokens = self.process_text(x)
        for w in tokens:
            self.d += 1
            self.vocab.add(w)
            if self.vocab.max_size == self.vocab.size:
                self._reduce_vocab()

            i = tokens.index(w)
            contexts = _get_contexts(i, self.window_size, tokens)
            for c in contexts:
                self.contexts.add(c)
                row = self.vocab.word2idx.get(w, 0)
                col = self.contexts.word2idx.get(c, 0)
                self.coocurence_matrix[row, col] += 1

    def learn_many(self, X: List[str], y=None, **kwargs) -> None:
        """Train a mini-batch of text features.

        Args:
            X: A list of sentence features.
            y: A series of target values, by default None.
        """
        for x in X:
            tokens = self.process_text(x)

            for w in tokens:

                self.d += 1
                self.vocab.add(w)

                if self.vocab.max_size == self.vocab.size:
                    self._reduce_vocab()

                i = tokens.index(w)
                contexts = _get_contexts(i, self.window_size, tokens)

                if self.reduced_emb_dim and w in self.vocab.word2idx:
                    self.modified_words.add(self.vocab.word2idx[w])

                for c in contexts:
                    self.contexts.add(c)
                    row = self.vocab.word2idx.get(w, 0)
                    col = self.contexts.word2idx.get(c, 0)
                    self.coocurence_matrix[row, col] += 1

    def _reduce_vocab(self) -> None:
        """Reduce the number of words in the vocabulary."""
        self.vocab.counter = self.vocab.counter - 1
        for idx, count in list(self.vocab.counter.items()):
            if count == 0:
                self.vocab.delete(idx)
                if self.reduced_emb_dim:
                    self.modified_words.discard(idx)

        indexes = np.array(list(self.vocab.free_idxs.copy()))
        self.coocurence_matrix[indexes, :] = 0.0
        if self.reduced_emb_dim:
            self.emb_matrix[indexes, :] = 0.0

    def get_embeddings(self, idxs: List[int]) -> np.ndarray:
        """Obtain a list of embedding given by a list of indexes.

        Args:
            idxs: List of indexes.
        Returns:
            List of embeddings vector.
        """
        words = [self.vocab.idx2word[idx] for idx in idxs]
        embs = np.array([self.transform_one(word) for word in words])
        return sparse.lil_matrix(embs)

    def transform_one(self, x: str) -> np.ndarray:
        """Obtain the vector embedding of a word.

        Args:
            x: word to obtain the embedding.

        Returns:
            The vector embedding of the word.
        """
        idx = self.vocab.word2idx[x]
        if idx in self.vocab.idx2word:
            contexts_ids = self.coocurence_matrix[idx].nonzero()[1]
            embedding = [0.0 for _ in range(self.context_size)]
            for cidx in contexts_ids:
                value = np.log2(
                    (self.d * self.coocurence_matrix[idx, cidx])
                    / (self.vocab.counter[idx] * self.contexts.counter[cidx])
                )
                embedding[cidx] = max(0.0, value)
        return embedding

    def _reduced_emb2dict(self) -> np.ndarray:
        if self.reduced_emb_dim:
            indexes = np.array(list(self.modified_words), dtype=float)
            embs = self.transformer.fit_transform(self.get_embeddings(indexes))

            self.emb_matrix[indexes] = embs

            self.modified_words.clear()

        embeddings = {}
        for word, idx in self.vocab.word2idx.items():
            embeddings[word] = self.emb_matrix[idx].toarray()
        return embeddings

    def vocab2dict(self) -> Dict[str, np.ndarray]:
        """Converts the vocabulary in a dictionary of embeddings.

        Returns:
            An dict where the words are the keys, and their values are the
                embedding vectors.
        """
        return self._reduced_emb2dict()


def _get_contexts(ind_word: int, w_size: int, tokens: List[str]) -> Tuple[str]:
    # to do: agregar try para check que es posible obtener los elementos de los tokens
    slice_start = ind_word - w_size if (ind_word - w_size >= 0) else 0
    slice_end = (
        len(tokens) if (ind_word + w_size + 1 >= len(tokens)) else ind_word + w_size + 1
    )
    first_part = tokens[slice_start:ind_word]
    last_part = tokens[ind_word + 1 : slice_end]
    contexts = tuple(first_part + last_part)
    return contexts
