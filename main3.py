from utils.data import TweetStream
from models import IWord2Vec
from evaluator import PeriodEvaluator


from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity


pe = PeriodEvaluator(
    TweetStream('/data/giturra/datasets/1e5tweets.txt'),
    IWord2Vec(sg=0),
    golden_dataset=fetch_MEN,
    eval_func=evaluate_similarity
)

pe.run(p=32000)