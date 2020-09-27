from text_classification.utils.utils import all_data_path as root_path
from text_classification.utils.utils import get_parameter_value
import torch
import random
from torchtext import data
from torchtext import datasets

SEED = 1024

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.detererministic = True


class IMDB(Dataset):
    def __init__(self, **kwargs):
        self.vocab_size = get_parameter_value(kwargs, vocab_size, 25000)
        self.batch_size= get_parameter_value(kwargs, batch_size, 32)
        self.device= get_parameter_value(kwargs, device, 'cpu')
        self.tokenizer= get_parameter_value(kwargs, tokenizer, 'spacy')
        self.train_valid_split_ratio = get_parameter_value(kwargs,
                                                           train_valid_split_ratio, 0.7)

        self.TEXT = data.Field(tokenize = tokenizer)
        self.LABEL = data.LabelField(dtype = torch.float)
        self.get_datasets()
        self.get_iterators(self.device, self.batch_size)


    def get_datasets(self):
        self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT,
                                                               self.LABEL,
                                                               root=root_path)
        self.train_data, self.valid_data = self.train_data.split(random_state=random.seed(SEED),
                                                                 split_ratio=self.train_valid_split_ratio)


    def get_iterators(self, self.device, self.batch_size):
        self.train_iterator, self.valid_iterator,\
            self.test_iterator = data.BucketIterator.splits(
                                                        (self.train_data,
                                                        self.valid_data,
                                                        self.test_data),
                                                        batch_size=self.batch_size,
                                                        device=self.device)


