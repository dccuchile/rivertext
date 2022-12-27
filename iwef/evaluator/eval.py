from typing import Callable

from torch.utils.data import DataLoader, IterableDataset

from iwef.models.base import IWVBase


class PeriodEvaluator:
    def __init__(
        self,
        dataset: IterableDataset,
        model: IWVBase,
        batch_size: int = 32,
        golden_dataset: Callable = None,
        eval_func=None,
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.model = model
        self.gold_relation = golden_dataset()
        self.evaluator = eval_func

    def run(self, p: int = 3200):
        c = 0
        for batch in self.dataloader:
            self.model.learn_many(batch)
            if c != 0 and c % p == 0:
                embs = self.model.vocab2dict()
                result = self.evaluator(
                    embs, self.gold_relation.X, self.gold_relation.y
                )
                print(result)
            c += len(batch)
