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


class IMDB():
    def __init__(self, **kwargs):
        self.vocab_size = get_parameter_value(kwargs, 'vocab_size', 25000)
        self.batch_size= get_parameter_value(kwargs, 'batch_size', 32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenize= get_parameter_value(kwargs, 'tokenize', 'spacy')
        self.train_valid_split_ratio = get_parameter_value(kwargs,
                                                           'train_valid_split_ratio', 0.7)
        self.embedding_vectors = get_parameter_value(kwargs,
                                                     'embedding_vectors')
        self.unk_init = get_parameter_value(kwargs, 'unk_initflag')
        self.include_lengths = get_parameter_value(kwargs, 'include_lengths', False)
        self.preprocessing = get_parameter_value(kwargs, 'preprocessing')
        self.init_token = get_parameter_value(kwargs, 'init_token')
        self.eos_token = get_parameter_value(kwargs, 'eos_token')
        self.pad_token = get_parameter_value(kwargs, 'pad_token')
        self.unk_token = get_parameter_value(kwargs, 'unk_token')
        self.batch_first = get_parameter_value(kwargs, 'batch_first', False)
        self.use_vocab = get_parameter_value(kwargs, 'use_vocab', True)
        self.vectors_cache = root_path
        if self.unk_init is True:
            self.unk_init = torch.Tensor.normal_
        else:
            self.unk_init = None
            self.vectors_cache = None
        #----
        self.TEXT = data.Field(tokenize=self.tokenize,
                               use_vocab=self.use_vocab,
                               batch_first=self.batch_first,
                               include_lengths=self.include_lengths,
                               preprocessing = self.preprocessing,
                               init_token = self.init_token,
                               eos_token = self.eos_token,
                               pad_token = self.pad_token,
                               unk_token = self.unk_token)
        self.LABEL = data.LabelField(dtype=torch.float)
        self.get_datasets()
        self.get_iterators(self.device, self.batch_size)

    def get_datasets(self):
        self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT,
                                                               self.LABEL,
                                                               root=root_path)
        if self.use_vocab:
            self.TEXT.build_vocab(self.train_data,
                                max_size=self.vocab_size,
                                vectors=self.embedding_vectors,
                                vectors_cache=self.vectors_cache,
                                unk_init = self.unk_init)
        self.LABEL.build_vocab(self.train_data)
        self.train_data, self.valid_data = self.train_data.split(random_state=random.seed(SEED),
                                                                 split_ratio=self.train_valid_split_ratio)


    def get_iterators(self, device, batch_size):
        self.train_iterator, self.valid_iterator,\
            self.test_iterator = data.BucketIterator.splits(
                                                        (self.train_data,
                                                        self.valid_data,
                                                        self.test_data),
                                                        batch_size=batch_size,
                                                        sort_within_batch = True,
                                                        device=device)


