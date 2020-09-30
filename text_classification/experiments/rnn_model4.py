from text_classification.datasets.imdb import IMDB
from text_classification.utils.metrics import binary_accuracy
from text_classification.models.bert import model
from text_classification.utils.utils import all_data_path, generate_bigrams
from text_classification.utils.utils import tokenize_and_cut, get_bert_preprocessing
from text_classification.utils.utils import get_bert_init_token, get_bert_pad_token
from text_classification.utils.utils import get_bert_eos_token, get_bert_unk_token

dataset_params = {
    'vocab_size': 25000,
    'batch_size': 32,
    'batch_first': True,
    'use_vocab': False,
    'tokenize': tokenize_and_cut,
    'preprocessing': get_bert_preprocessing(),
    'train_valid_split_ratio': 0.7,
    'init_token': get_bert_init_token,
    'eos_token': get_bert_eos_token,
    'pad_token': get_bert_pad_token,
    'unk_token': get_bert_unk_token
}

imdb_data = IMDB(**dataset_params)

model_params = {
    'hidden_dim'L 256,
    'output_dim': 1,
    'n_layers': 2,
    'bidirectional': True,
    'dropout': 0.5,
    'evaluation_metric': binary_accuracy,
    'output_model_path': all_data_path,
    'data': imdb_data,
}

clf = model(**model_params)
clf.fit()
clf.predict()
