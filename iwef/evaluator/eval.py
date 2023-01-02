from typing import Callable, Dict

import numpy as np
from torch.utils.data import DataLoader, IterableDataset

from iwef.models.base import IWVBase


class PeriodEvaluator:
    """_summary_"""

    def __init__(
        self,
        dataset: IterableDataset,
        model: IWVBase,
        batch_size: int = 32,
        golden_dataset: Callable = None,
        eval_func: Callable[[Dict, np.ndarray, np.ndarray], int] = None,
    ):
        """_summary_

        Args:
            dataset: _description_
            model: _description_
            batch_size: _description_, by default 32
            golden_dataset: _description_, by default None
            eval_func: _description_, by default
                Callable[[Dict, np.ndarray, np.ndarray], int]
        """
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.model = model
        self.gold_relation = golden_dataset()
        self.evaluator = eval_func

    def run(self, p: int = 3200):
        """_summary_

        Args:
            p: _description_, by default 3200
        """
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
