import torch
import torch.nn as nn
import torch.optim as optim
import time
from text_classification.utils.utils import train, evaluate
from text_classification.utils.utils import get_parameter_value, epoch_time


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        #text = [sent len, batch size]
        embedded = self.embedding(text)
        #embedded = [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 0, 2)
        #embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        #pooled = [batch size, embedding_dim]
        return self.fc(pooled)


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
        self.padidx = self.data.vocab.stoi[TEXT.pad_token]
        self.device = self.data.device
        self.train_iterator = self.data.train_iterator
        self.valid_iterator = self.data.valid_iterator
        self.test_iterator = self.data.test_iterator
        self.model = FastText(self.input_dim,
                        self.embedding_dim,
                        self.hidden_dim,
                        self.output_dim,
                        self.padidx
                        )
        self.pretrained_embeddings = self.data.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(self.pretrained_embeddings)
        UNK_IDX = self.data.TEXT.vocab.stoi[TEXT.unk_token]

        self.model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        self.optimizer = optim.Adam(model.parameters())
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
