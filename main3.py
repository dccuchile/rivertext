from evaluator import PeriodEvaluator
from models import WordContextMatrix
from utils.data import TweetStream
from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity

pe = PeriodEvaluator(
    TweetStream("/data/giturra/datasets/1e5tweets.txt"),
    # IWord2Vec(sg=0),
    WordContextMatrix(1_000_000, 3, 500, emb_size=100),
    golden_dataset=fetch_MEN,
    eval_func=evaluate_similarity,
)

pe.run(p=32000)
# print(pe.model.vocab.word2idx.keys())
pe.model.vocab2dict()
