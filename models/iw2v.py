from .base import IncrementalWordVector
from .iword2vec import (
    SkipGram, CBOW, PrepSkipGram, PrepCbow
)

from torch.optim import SparseAdam


from tqdm import tqdm


class IWord2Vec(IncrementalWordVector):

    def __init__(
        self,
        batch_size=32,
        vocab_size: int=1_000_000,
        emb_size=100, 
        unigram_table_size: int=100_000_000,
        window_size: int=5, 
        alpha: float=0.75,
        subsampling_threshold:float =1e-3,
        neg_samples_sum: int=10, 
        sg=1, 
        lr=0.025,
        device=None, 
        optimizer=SparseAdam,
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


        self.neg_sample_num = neg_samples_sum
        self.sg = sg

        if sg:
            self.model_name =  'SG'
            self.model = SkipGram(self.vocab_size, emb_size)
            self.prep = PrepSkipGram(
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
            self.model_name = 'CBOW'
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

    def get_embedding(self, word):
        ...
    
    def vocab2dict(self):
        embeddings = {}
        for word in tqdm(self.prep.vocab.word2idx.keys()):
            embeddings[word] = self.transform_one(word)
        return embeddings 

    def transform_one(self, x):
        word_idx = self.prep.vocab[x]
        return self.model.get_embedding(word_idx)
    
    def learn_one(self, x, **kwargs):
        tokens = self.process_text(x)
        batch = self.prep(tokens)
        targets = batch[0].to(self.device)
        contexts = batch[1].to(self.device)
        neg_samples = batch[2].to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(targets, contexts, neg_samples)
        loss.backward()
        self.optimizer.step()

    def learn_many(self, X, y=None, **kwargs):
        tokens = list(map(self.process_text, X))
        batch = self.prep(tokens)
        targets = batch[0].to(self.device)
        contexts = batch[1].to(self.device)
        neg_samples = batch[2].to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(targets, contexts, neg_samples)
        loss.backward()
        self.optimizer.step()
        