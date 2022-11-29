from utils import TweetStream
from torch.utils.data import DataLoader

from models import WordContextMatrix

from tqdm import tqdm

ts = TweetStream(
    '/data/giturra/datasets/1e5tweets.txt'
)

from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity


wcm = WordContextMatrix(10000, 3, 500)

dataloader = DataLoader(ts, batch_size=32)

men = fetch_MEN()

for batch in tqdm(dataloader):
    wcm.learn_many(batch)


    
embs = wcm.reduced_emb2dict()
print(evaluate_similarity(embs, men.X, men.y))

