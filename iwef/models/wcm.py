""""""
from typing import Callable, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import IncrementalPCA

from iwef.models.base import IWVBase
from iwef.utils import Context, Vocab


class WordContextMatrix(IWVBase):
    """_summary_

    Parameters
    ----------
    IWVBase : _type_
        _description_
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
        preprocessor=None,
        tokenizer: Callable[[str], List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        """_summary_

        Parameters
        ----------
        vocab_size : int
            _description_
        window_size : int
            _description_
        context_size : int
            _description_
        emb_size : int, optional
            _description_, by default 300
        reduce_emb_dim : bool, optional
            _description_, by default True
        on : str, optional
            _description_, by default None
        strip_accents : bool, optional
            _description_, by default True
        lowercase : bool, optional
            _description_, by default True
        preprocessor : _type_, optional
            _description_, by default None
        tokenizer : Callable[[str], List[str]], optional
            _description_, by default None
        ngram_range : Tuple[int, int], optional
            _description_, by default (1, 1)
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
        """_summary_

        Parameters
        ----------
        x : str
            _description_
        """
        tokens = self.process_text(x)
        for w in tokens:
            self.d += 1
            self.vocab.add(w)
            if self.vocab.max_size == self.vocab.size:
                self.reduce_vocab()

            i = tokens.index(w)
            contexts = _get_contexts(i, self.window_size, tokens)
            for c in contexts:
                self.contexts.add(c)
                row = self.vocab.word2idx.get(w, 0)
                col = self.contexts.word2idx.get(c, 0)
                self.coocurence_matrix[row, col] += 1

    def learn_many(self, X: List[str], y=None, **kwargs) -> None:
        """_summary_

        Parameters
        ----------
        X : List[str]
            _description_
        y : _type_, optional
            _description_, by default None
        """
        for x in X:
            tokens = self.process_text(x)

            for w in tokens:

                self.d += 1
                self.vocab.add(w)

                if self.vocab.max_size == self.vocab.size:
                    self.reduce_vocab()

                i = tokens.index(w)
                contexts = _get_contexts(i, self.window_size, tokens)

                if self.reduced_emb_dim and w in self.vocab.word2idx:
                    self.modified_words.add(self.vocab.word2idx[w])

                for c in contexts:
                    self.contexts.add(c)
                    row = self.vocab.word2idx.get(w, 0)
                    col = self.contexts.word2idx.get(c, 0)
                    self.coocurence_matrix[row, col] += 1

    def reduce_vocab(self) -> None:
        """_summary_"""
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

    def tokens2idxs(self, tokens: List[str]) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        tokens : List[str]
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        idxs = []
        for token in tokens:
            idxs.append(self.vocab.word2idx.get(token, -1))
        return idxs

    def get_embeddings(self, idxs: List[int]) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        idxs : List[int]
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        words = [self.vocab.idx2word[idx] for idx in idxs]
        embs = np.array([self.transform_one(word) for word in words])
        return sparse.lil_matrix(embs)

    def transform_one(self, x: str) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        x : str
            _description_

        Returns
        -------
        np.ndarray
            _description_
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

    def vocab2dict(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        return self._reduced_emb2dict()

    def vocab2matrix(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        mat = np.empty((self.vocab_size, self.context_size))
        for word, idx in self.vocab.word2idx.items():
            mat[idx] = self.transform_one(word)
        return sparse.lil_matrix(mat)


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
