import abc

from river.base.transformer import Transformer
from river.feature_extraction.vectorize import VectorizerMixin
from river.utils import numpy2dict


class IncrementalWordVector(Transformer, VectorizerMixin):

    def __init__(
        self,
        vocab_size,
        vector_size,
        window_size,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1),
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
    def learn_many(self, X, y=None, **kwargs):
        ...
    

    # @abc.abstractmethod
    # def get_embedding(self, word):
    #     ...
    
    def embedding2dict(self, word):
        emb = self.get_embedding(word)
        return numpy2dict(emb)
    
    def vocab2dict(self):
        embeddings = {}
        for word in self.vocab.word2idx.keys():
            embeddings[word] = self.get_embedding(word)
        return embeddings