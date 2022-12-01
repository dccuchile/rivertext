# flake8: noqa
# from iword2vec import PrepCbow

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import TweetStream

# from models import WordContextMatrix


ts = TweetStream("/data/giturra/datasets/1e5tweets.txt")

from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity

# wcm = WordContextMatrix(10000, 3, 500)

dataloader = DataLoader(ts, batch_size=32)

men = fetch_MEN()

# for batch in tqdm(dataloader):
#     wcm.learn_many(batch)


from models import IWord2Vec

iw2v = IWord2Vec(emb_size=100, device="cuda:1")

for batch in tqdm(dataloader):
    iw2v.learn_many(batch)

embs = iw2v.vocab2dict()
print(evaluate_similarity(embs, men.X, men.y))
