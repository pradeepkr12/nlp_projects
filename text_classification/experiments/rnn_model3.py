from text_classification.datasets.imdb import IMDB
from text_classification.utils.metrics import binary_accuracy
from text_classification.models.rnn import RNN2
from text_classification.models.model import model
from text_classification.utils.utils import all_data_path, generate_bigrams

dataset_params = {
    'vocab_size': 25000,
    'batch_size': 32,
    'tokenizer': 'spacy',
    'train_valid_split_ratio': 0.7,
    'unk_initflag': True,
    'preprocessing': generate_bigrams,
    'embedding_vector': 'glove.6B.100d'
}
imdb_data = IMDB(**dataset_params)

model_params = {
    'embedding_dim': 100,
    'hidden_dim': 256,
    'output_dim': 1,
    'n_layers': 2,
    'evaluation_metric': binary_accuracy,
    'output_model_path': all_data_path,
    'data': imdb_data,
}

clf = model(**model_params)
clf.fit()
clf.predict()
