from typing import Iterator, List

from torch.utils.data import IterableDataset


class TweetStream(IterableDataset):
    def __init__(self, filename: str):
        """_summary_

        Args:
            filename: _description_
        """
        self.filename = filename

    def preprocess(self, text: str) -> List[str]:
        """_summary_

        Args:
            text: _description_

        Returns:
            _description_
        """

        tweet = text.rstrip("\n")
        return tweet

    def __iter__(self) -> Iterator:
        """_summary_

        Yields:
            _description_
        """
        file_itr = open(self.filename, encoding="utf-8")
        mapped_itr = map(self.preprocess, file_itr)
        return mapped_itr
