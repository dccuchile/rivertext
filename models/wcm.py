import numpy as np
from scipy import sparse
from river.utils import numpy2dict

from utils import Vocab, Context
from .base import IncrementalWordVector

from sklearn.decomposition import IncrementalPCA


class WordContextMatrix(IncrementalWordVector):

    def __init__(
        self, 
        vocab_size,  
        window_size,
        context_size,
        emb_size=300,
        reduce_emb_dim=True,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1)
    ):
        super().__init__(
            vocab_size,
            emb_size,
            window_size,
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range
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

            self.transformer = IncrementalPCA(n_components=self.vector_size, batch_size=300)

    
    def learn_one(self, x, **kwargs):
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

    
    def learn_many(self, X, y=None, **kwargs):
        
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
                


    def reduce_vocab(self):
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


    def tokens2idxs(self, tokens):
        idxs = []
        for token in tokens:
            idxs.append(self.vocab.word2idx.get(token, -1))
        return idxs
    
    def get_embeddings(self, idxs):
        words = [self.vocab.idx2word[idx] for idx in idxs]
        embs = np.array([self.transform_one(word) for word in words])
        return sparse.lil_matrix(embs)


    def transform_one(self, x):
        idx = self.vocab.word2idx[x]
        if idx in self.vocab.idx2word:
            contexts_ids = self.coocurence_matrix[idx].nonzero()[1]
            embedding = [0.0 for _ in range(self.context_size)]
            for cidx in contexts_ids:
                value = np.log2(
                    (self.d * self.coocurence_matrix[idx, cidx]) / (self.vocab.counter[idx] * self.contexts.counter[cidx])
                )
                embedding[cidx] = max(0.0, value)
        return embedding     

    def reduced_emb2dict(self):
        if self.reduced_emb_dim:
            indexes = np.array(list(self.modified_words), dtype=float)
            embs = self.transformer.fit_transform(self.get_embeddings(indexes))

            self.emb_matrix[indexes] = embs

            self.modified_words.clear()
        
        embeddings = {}
        for word, idx in self.vocab.word2idx.items():
            embeddings[word] = self.emb_matrix[idx].toarray()
        return embeddings


    
    def vocab2dict(self):
        embeddings = {}
        for word in self.vocab.word2idx.keys():
            embeddings[word] = self.transform_one(word)
        return embeddings

    def vocab2matrix(self):
        mat = np.empty((self.vocab_size, self.context_size))
        for word, idx in self.vocab.word2idx.items():
            mat[idx] = self.transform_one(word)
        return sparse.lil_matrix(mat)



def _get_contexts(ind_word, w_size, tokens):
    # to do: agregar try para check que es posible obtener los elementos de los tokens
    slice_start = ind_word - w_size if (ind_word - w_size >= 0) else 0
    slice_end = len(tokens) if (ind_word + w_size + 1 >= len(tokens)) else ind_word + w_size + 1
    first_part = tokens[slice_start: ind_word]
    last_part = tokens[ind_word + 1: slice_end]
    contexts = tuple(first_part + last_part)
    return contexts