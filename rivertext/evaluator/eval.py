import os
import json
from typing import Callable, Dict

import numpy as np
from torch.utils.data import DataLoader, IterableDataset

from rivertext.models.base import IWVBase


class PeriodicEvaluator:
    """Periodic Evaluation assesses the entire incremental word embeddingsmodel's
    performance using an intrinsic NLP task-related test dataset after a set number, p,
    of instances, have been processed and trained. This allows for the continuous
    evaluation of the model's accuracy and helps identify improvement areas."""

    def __init__(
        self,
        dataset: IterableDataset,
        model: IWVBase,
        p: int = 32,
        golden_dataset: Callable = None,
        eval_func: Callable[[Dict, np.ndarray, np.ndarray], int] = None,
        path_output_file: str = None,
    ):
        """Create a instance of PeriodicEvaluator class.

        Args:
            dataset: Stream to train.
            model: Model to train.
            batch_size: batch size for the dataloader, by default 32
            golden_dataset: Golden dataset relations, by default None
            eval_func: Function evaluator acording to the golden dataset, by default
            None.
        """
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=p)
        self.model = model
        self.gold_relation = golden_dataset()
        self.evaluator = eval_func
        self.path_output_file = path_output_file

        if not self.path_output_file.endswith(".json"):
            raise ValueError(
                f"the extension file must be an JSON, but you got: \
                {self.path_output_file}."
            )

    def run(self, p: int = 3200):
        """Algorithm executes periodic assessments of the entire
        model every p instances, providing continuous evaluation and identification of
        areas for improvement.

        Args:
            p: Number of instances to process before evaluating the model,
                by default 3200.
        """
        c = 0
        for batch in self.dataloader:
            self.model.learn_many(batch)
            if c != 0 and c % p == 0:
                embs = self.model.vocab2dict()
                result = self.evaluator(
                    embs, self.gold_relation.X, self.gold_relation.y
                )
                self._save_result(result)
            c += len(batch)

    def _save_result(self, result: float):
        if self.path_output_file is not None and not os.path.exists(
            self.path_output_file
        ):
            with open(self.path_output_file, "w", encoding="utf-8") as writer:
                json.dump(
                    {"model_name": self.model.model_name, "values": [result]}, writer
                )
        else:
            with open(self.path_output_file, encoding="utf-8") as reader:
                data = json.load(reader)
                data["values"].append(result)

            with open(self.path_output_file, "w", encoding="utf-8") as writer:
                json.dump(data, writer)
