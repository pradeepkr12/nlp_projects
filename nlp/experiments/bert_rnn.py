import torch
from torchtext import data
from torchtext.vocab import Vectors
import torch.optim as optim
import torch.nn as nn
import time
import logging
from transformers import BertTokenizer, BertModel

from nlp.datasets import IMDB
from nlp.models.bert import BERTGRUSentiment
from nlp.models.utils import train, evaluate, epoch_time
from nlp.utils import mylogger

MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64
IMDB_DATAPATH = "/Users/pradeepkumarmahato/pradeep/nlp/torch-data/aclImdb"
SPACY_LANGUAGE = None
PRETRAINED_MODEL_PATH = None
BERT_PRETRAINED_TOKENIZER = "/Users/pradeepkumarmahato/pradeep/nlp/bert_bin/bert-base-uncased-vocab.txt"
BERT_PRETRAINED_TOKENIZER = 'bert-base-uncased'

try:
    import spacy
    nlp = spacy.load(SPACY_LANGUAGE)
except:
    print ("Spacy language missing")
    import os
    os.system(f"python -m spacy download {SPACY_LANGUAGE}")



def run_experiment():

    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_TOKENIZER)
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length - 2]
        return tokens

    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id

    imdb = IMDB.IMDB_dataset(IMDB_DATAPATH,
                             include_lengths=False,
                             batch_first=True,
                             tokenize=tokenize_and_cut,
                             preprocessing=tokenizer.convert_tokens_to_ids,
                             init_token=init_token_idx,
                             eos_token=eos_token_idx,
                             pad_token=pad_token_idx,
                             unk_token=unk_token_idx
                             )
    train_data, valid_data, test_data = imdb.get_data(validation=True)

    # preprocess the data
    TEXT = imdb.TEXT
    LABEL = imdb.LABEL
    LABEL.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                        (train_data,
                                                        valid_data,
                                                        test_data),
                                                        batch_size=BATCH_SIZE,
                                                        device=device)
    # modelling

    bert = BertModel.from_pretrained('bert-base-uncased')
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    model = BERTGRUSentiment(bert,
                             HIDDEN_DIM,
                             OUTPUT_DIM,
                             N_LAYERS,
                             BIDIRECTIONAL,
                             DROPOUT)

    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    best_valid_loss = float('inf')
    N_EPOCHS = 5
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
