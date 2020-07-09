import torch
from torchtext import data
from torchtext import datasets

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.detererministic = True


class IMDB_dataset:
    '''
    IMDB Dataset, using torchtext dataset to get the data
    '''
    def __init__(self, imdb_datapath=None):
        self.TEXT = data.Field(tokenize='spacy',
                               tokenizer_language='en_core_web_sm')
        self.LABEL = data.LabelField(dtype = torch.float)
        self.imdb_datapath = imdb_datapath
        self.imdb = datasets.IMDB(self.imdb_datapath, self.TEXT, self.LABEL)
        self.train_data, self.test_data = self.imdb.splits(
                                            path=self.imdb_datapath,
                                            root=None,
                                            text_field=self.TEXT,
                                            label_field=self.LABEL,
                                            train='train', test='test')

    def get_data(self):
        '''
        Returns train, test dataset
        '''
        return self.train_data, self.test_data
