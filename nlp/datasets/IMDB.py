import torch
import random
from torchtext import data
from torchtext import datasets

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.detererministic = True


class IMDB_dataset:
    '''
    IMDB Dataset, using torchtext dataset to get the data
    '''
    def __init__(self, imdb_datapath=None, **kwargs):
        self.TEXT = data.Field(kwargs)
        self.LABEL = data.LabelField(dtype=torch.float)
        self.imdb_datapath = imdb_datapath
        if imdb_datapath is not None:
            self.imdb = datasets.IMDB(self.imdb_datapath, self.TEXT, self.LABEL)
            self.train_data, self.test_data = self.imdb.splits(
                                                path=self.imdb_datapath,
                                                root=None,
                                                text_field=self.TEXT,
                                                label_field=self.LABEL,
                                                train='train', test='test')
        else:
            self.train_data, self.test_data = datasets.IMDB.splits(
                                                text_field=self.TEXT,
                                                label_field=self.LABEL,
                                                train='train', test='test')


    def get_data(self, train=None, test=None, validation=None):
        '''
        Returns train, test dataset
        '''
        if validation is None:
            return self.train_data, self.test_data
        else:
            # split the train into train and validation data and send
            train_data, validaton_data = self.train_data.split(
                                            random_state=random.seed(SEED))
            return train_data, validaton_data, self.test_data
