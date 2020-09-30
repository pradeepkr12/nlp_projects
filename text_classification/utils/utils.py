from pathlib import Path
from transformers import BertTokenizer

import torch

all_data_path = Path(__file__).resolve().parent.parent/"data"

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def get_parameter_value(value_dict, key, default_value=None):
    value = value_dict.get(key)
    if value is None:
        if default_value is None:
            return None
        else:
            return default_value
    return value


def train(model, iterator, optimizer, criterion, evaluation_metric):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        if isinstance(batch.text, torch.Tensor):
            predictions = model(batch.text).squeeze(1)
        else:
            predictions = model(*batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = evaluation_metric(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, evaluation_metric):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch.text, torch.Tensor):
                predictions = model(batch.text).squeeze(1)
            else:
                predictions = model(*batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = evaluation_metric(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_bert_tokenizer(model='bert-base-uncased'):
    return BertTokenizer.from_pretrained(model)

def get_bert_preprocessing():
    import pdb;pdb.set_trace()
    tokenizer = get_bert_tokenizer()
    return tokenizer.convert_tokens_to_ids

def tokenize_and_cut(sentence, model='bert-base-uncased'):
    tokenizer = get_bert_tokenizer()
    max_input_length = tokenizer.max_model_input_sizes[model]
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

def get_bert_init_token():
    tokenizer = get_bert_tokenizer()
    return tokenizer.cls_token_id

def get_bert_eos_token():
    tokenizer = get_bert_unk_token()
    return tokenizer.eos_token_idx

def get_bert_pad_token():
    tokenizer = get_bert_unk_token()
    return tokenizer.pad_token_idx

def get_bert_unk_token():
    tokenizer = get_bert_unk_token()
    return tokenizer.unk_token_idx
