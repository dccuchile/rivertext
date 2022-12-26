import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Word2Vec(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(Word2Vec, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension

        # syn0: embedding for input words
        # syn1: embedding for output words
        self.syn0 = nn.Embedding(emb_size, emb_dimension, sparse=True, padding_idx=0)
        self.syn1 = nn.Embedding(emb_size, emb_dimension, sparse=True, padding_idx=0)

        init_range = 0.5 / self.emb_dimension
        init.uniform_(self.syn0.weight.data, -init_range, init_range)
        init.constant_(self.syn1.weight.data, 0)
        self.syn0.weight.data[0, :] = 0

    def forward(self, pos_u, pos_v, neg_v):
        raise NotImplementedError()

    def get_embedding(self, idx):
        return (self.syn0.weight[idx] + self.syn1.weight[idx]).cpu().detach().numpy()


class SG(Word2Vec):
    def __init__(self, emb_size, emb_dimension):
        super(SG, self).__init__(emb_size, emb_dimension)

    def forward(self, target, context, negatives):
        t = self.syn0(target)
        c = self.syn1(context)

        score = torch.mul(t, c).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        n = self.syn1(negatives)
        neg_score = torch.bmm(n, t.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))


class CBOW(Word2Vec):
    def __init__(self, emb_size, emb_dimension, cbow_mean=True):
        super(CBOW, self).__init__(emb_size, emb_dimension)
        self.cbow_mean = cbow_mean

    def forward(self, target, context, negatives):
        t = self.syn1(target)
        c = self.syn0(context)

        # Mean of context vector without considering padding idx (0)
        if self.cbow_mean:
            mean_c = torch.sum(c, dim=1) / torch.sum(context != 0, dim=1, keepdim=True)
        else:
            mean_c = c.sum(dim=1)

        score = torch.mul(t, mean_c).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        n = self.syn1(negatives)
        neg_score = torch.bmm(n, mean_c.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))
