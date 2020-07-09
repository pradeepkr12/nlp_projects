import torch
from torchtext import data
import torch.optim as optim
import torch.nn as nn


from nlp.datasets import IMDB
from nlp.models.rnn import RNN
from nlp.models.utils import train, evaluate, epoch_time

MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64


def run_experiment():
    imdb_datapath = "/Users/pradeepkumarmahato/pradeep/nlp/torch-data/aclImdb"
    spacy_language = "en_core_web_sm"
    imdb = IMDB.IMDB_dataset(imdb_datapath, spacy_language)
    train_data, valid_data, test_data = imdb.get_data(validation=True)

    # preprocess the data

    TEXT = imdb.TEXT
    LABEL = imdb.LABEL
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
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
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
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