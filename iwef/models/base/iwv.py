import abc
from typing import Callable, Dict, List, Tuple

import numpy as np
from river.base.transformer import Transformer
from river.feature_extraction.vectorize import VectorizerMixin


class IncrementalWordVector(Transformer, VectorizerMixin):
    def __init__(
        self,
        vocab_size: int,
        vector_size: int,
        window_size: int,
        on: str = None,
        strip_accents: bool = True,
        lowercase: bool = True,
        preprocessor=None,
        tokenizer: Callable[[str], List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
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
        raise NotImplementedError

    @abc.abstractmethod
    def vocab2dict(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError
