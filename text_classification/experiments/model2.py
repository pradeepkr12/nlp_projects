
TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)


from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)
