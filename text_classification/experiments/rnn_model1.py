from text_classification.datasets.imbd import IMDB
from text_classification.utils.metrics import binary_accuracy
from text_classification.models.rnn import model as rnn_model


dataset_params = {
    'vocab_size': 25000,
    'batch_size': 32,
    'device': 'cpu',
    'tokenizer': 'spacy',
    'train_valid_split_ratio': 0.7
}
imdb_data = IMDB(**dataset_params)

model_params = {
    'input_dim': len(imdb_data.TEXT.vocab),
    'embedding_dim': 100,
    'hidden_dim': 256,
    'output_dim': 1,
    'device': imdb_data.device,
    'evaluation_metric': binary_accuracy,

}

clf = rnn_model(**model_params)
clf.fit()
clf.predict()
