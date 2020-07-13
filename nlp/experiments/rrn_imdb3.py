import torch
from torchtext import data
from torchtext.vocab import Vectors
import torch.optim as optim
import torch.nn as nn
import time
import logging

from nlp.datasets import IMDB
from nlp.models.fasttext import FastText
from nlp.models.utils import train, evaluate, epoch_time
from nlp.utils import mylogger

MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64
IMDB_DATAPATH = "/Users/pradeepkumarmahato/pradeep/nlp/torch-data/aclImdb"
SPACY_LANGUAGE = "en_core_web_sm"
PRETRAINED_MODEL_PATH = "/Users/pradeepkumarmahato/pradeep/nlp/torch-data/glove/glove.6B/glove.6B.300d.txt"


try:
    import spacy
    nlp = spacy.load(SPACY_LANGUAGE)
except:
    print ("Spacy language missing")
    import os
    os.system(f"python -m spacy download {SPACY_LANGUAGE}")


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def run_experiment():
    imdb = IMDB.IMDB_dataset(IMDB_DATAPATH, SPACY_LANGUAGE,
                             include_lengths=False,
                             preprocessing=generate_bigrams)
    train_data, valid_data, test_data = imdb.get_data(validation=True)

    # preprocess the data
    TEXT = imdb.TEXT
    LABEL = imdb.LABEL
    # TODO
    if PRETRAINED_MODEL_PATH is not None:
        pretrained_weights = Vectors(name=PRETRAINED_MODEL_PATH,
                                     unk_init=torch.Tensor.normal_)

        TEXT.build_vocab(train_data,
                         vectors=pretrained_weights,
                         max_size=MAX_VOCAB_SIZE)
    else:
        TEXT.build_vocab(train_data,
                         max_size=MAX_VOCAB_SIZE,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                        (train_data,
                                                        valid_data,
                                                        test_data),
                                                        batch_size=BATCH_SIZE,
                                                        device=device)
    # modelling
    N_EPOCHS = 5
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    OUTPUT_DIM = 1
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

    # set the pretrained weights
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
