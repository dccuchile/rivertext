from torch.utils.data import IterableDataset


class TweetStream(IterableDataset):

    def __init__(self, filename):
        self.filename = filename
            
    def preprocess(self, text):
        tweet = text.rstrip('\n')
        return tweet
    
    def __iter__(self):
        file_itr = open(self.filename, encoding='utf-8')
        mapped_itr = map(self.preprocess, file_itr)
        return mapped_itr
