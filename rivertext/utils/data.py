"""Module for adding new iterable datasets."""

from typing import Iterator, List

from torch.utils.data import IterableDataset


class TweetStream(IterableDataset):
    """An Iterable Dataset extends the Iterable class from the PyTorch package.

    The Iterable Dataset class is designed to process big volumes of tweets that
        do not necessarily fit in memory.

    The tweets are expected to be separated by a line break through the file on disk.

    Examples:
        >>> from rivertext.utils import TweetStream
        >>> ts = TweetStream("/path/to/tweets.txt")
        >>> dataloader = DataLoader(ts, batch_size=1)
        >>> for batch in dataloader:
        ...     print(batch)
        >>> "hello how are you?"
        >>> "This is tweet example?"

    """

    def __init__(self, filename: str):
        """An instance of TweetStream class.

        Args:
            filename: path to the tweets file in the disk.


        """
        self.filename = filename

    def preprocess(self, text: str) -> List[str]:
        """Remove the whitespace for the current tweet.

        Args:
            text: tweet to remove whitespace.

        Returns:
            A String without whitespaces.
        """

        tweet = text.rstrip("\n")
        return tweet

    def __iter__(self) -> Iterator:
        """Take some tweets from the file on the disk, creating a generator.

        Examples:
            >>> from rivertext.utils import TweetStream
            >>> ts = TweetStream("/path/to/tweets.txt")
            >>> next(ts)
            >>> "hello how are you?"
            >>> next(ts)
            >>> "This is tweet example?"

        Yields:
            A generator of tweets.
        """
        file_itr = open(self.filename, encoding="utf-8")
        mapped_itr = map(self.preprocess, file_itr)
        return mapped_itr

    def __len__(self):
        return len(self.filename)
