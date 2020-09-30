import torch
import torch.nn as nn
import torch.optim as optim
import time
from text_classification.utils.utils import train, evaluate
from text_classification.utils.utils import get_parameter_value, epoch_time


class RNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        #text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)


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
        self.n_layers = get_parameter_value(kwargs, 'n_layers')
        self.bidirectional = get_parameter_value(kwargs, 'bidirectional')
        self.dropout = get_parameter_value(kwargs, 'dropout')
        self.data = get_parameter_value(kwargs, 'data')
        if self.data is None:
            raise Exception("Training data is None, please check")
        self.input_dim = len(self.data.TEXT.vocab)
        self.padidx = self.data.TEXT.vocab.stoi[self.data.TEXT.pad_token]
        self.device = self.data.device
        self.train_iterator = self.data.train_iterator
        self.valid_iterator = self.data.valid_iterator
        self.test_iterator = self.data.test_iterator
        # --- model init
        self.model = RNN(self.input_dim,
                        self.embedding_dim,
                        self.hidden_dim,
                        self.output_dim,
                        self.n_layers,
                        self.bidirectional,
                        self.dropout,
                        self.padidx
                        )
        self.pretrained_embeddings = self.data.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(self.pretrained_embeddings)
        UNK_IDX = self.data.TEXT.vocab.stoi[self.data.TEXT.unk_token]
        PAD_IDX = self.padidx
        self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.embedding_dim)
        self.model.embedding.weight.data[PAD_IDX] = torch.zeros(self.embedding_dim)
        # -----
        self.optimizer = optim.Adam(self.model.parameters())
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
        print(f"Training in {self.device}")
        for epoch in range(self.n_epochs):
            start_time = time.time()
            import pdb;pdb.set_trace()
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
                                                 self.criterion,
                                                 self.evaluation_metric)

        print(f'Test Loss: {self.test_loss:.3f} | Test Acc: {self.test_acc*100:.2f}%')
