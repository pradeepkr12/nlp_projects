import torch
import torch.nn as nn
import torch.optim as optim
import time
from text_classification.utils.utils import train, evaluate
from text_classification.utils.utils import get_parameter_value, epoch_time

from transformers import BertTokenizer, BertModel

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        #text = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        #embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)
        #hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        #hidden = [batch size, hid dim]
        output = self.out(hidden)
        #output = [batch size, out dim]
        return output

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
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.model = BERTGRUSentiment(self.bert,
                                 self.hidden_dim,
                                 self.output_dim,
                                 self.n_layers,
                                 self.bidirectional,
                                 self.dropout,
                                 )
        for name, param in self.model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
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
            # import pdb;pdb.set_trace()
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
