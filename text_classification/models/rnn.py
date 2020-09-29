from text_classification.utils.utils import get_parameter_value, epoch_time

import torch
import torch.nn as nn
import torch.optim as optim
import time


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))


class model:
    def __init__(self, **kwargs):
        self.embedding_dim = get_parameter_value(kwargs, 'embedding_dim')
        self.hidden_dim = get_parameter_value(kwargs, 'hidden_dim')
        self.output_dim = get_parameter_value(kwargs, 'output_dim')
        self.n_epochs = get_parameter_value(kwargs, 'n_epochs', 10)
        self.evaluation_metric = get_parameter_value(kwargs,
                                                     'evaluation_metric')
        self.output_model_path = get_parameter_value(kwargs,
                                                     'output_model_path')

        self.data = get_parameter_value(kwargs, 'data')
        if self.data is None:
            raise Exception("Training data is None, please check")
        self.input_dim = len(self.data.TEXT.vocab)
        self.device = self.data.device
        self.train_iterator = self.data.train_iterator
        self.valid_iterator = self.data.valid_iterator
        self.test_iterator = self.data.test_iterator
        self.model = RNN(self.input_dim,
                         self.embedding_dim,
                         self.hidden_dim,
                         self.output_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()
        # change deivice
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def total_paramters(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)

    def fit(self):
        self.best_valid_loss = float('inf')
        self.best_valid_acc = 0
        self.output_model_filepath = self.output_model_path/"rnn_model_weights.pt"
        for epoch in range(self.n_epochs):
            start_time = time.time()
            train_loss, train_acc = train(self.model,
                                          self.train_iterator,
                                          self.optimizer,
                                          self.criterion,
                                          self.evaluation_metric)
            valid_loss, valid_acc = evaluate(self.model,
                                             self.valid_iterator,
                                             self.criterion,
                                             self.evaluation_metric)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), self.output_model_filepath)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    def predict(self, saved_model=None):
        if saved_model is None:
            saved_model = self.output_model_filepath
        self.model.load_state_dict(torch.load(saved_model))
        self.test_loss, self.test_acc = evaluate(self.model,
                                                 self.test_iterator,
                                                 self.criterion)

        print(f'Test Loss: {self.test_loss:.3f} | Test Acc: {self.test_acc*100:.2f}%')


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
            predictions = model(*batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = evaluation_metric(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
