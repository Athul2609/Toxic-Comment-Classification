import torch
from torch import nn
import config

class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size=hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), padding_idx=40000)
        self.bilstm1 = nn.LSTM(input_size=embedding_matrix.shape[1], hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)

        x=x.float()
        h0 = torch.zeros(2, config.BATCH_SIZE, self.hidden_size).to(config.DEVICE)
        c0 = torch.zeros(2, config.BATCH_SIZE, self.hidden_size).to(config.DEVICE)

        lstm_out1, _ = self.bilstm1(x, (h0, c0))

        h1 = torch.zeros(2, config.BATCH_SIZE, self.hidden_size).to(config.DEVICE)
        c1 = torch.zeros(2, config.BATCH_SIZE, self.hidden_size).to(config.DEVICE)

        lstm_out2, _ = self.bilstm2(lstm_out1, (h1, c1))
        lstm_out = lstm_out2[:, -1, :]  # Use the final hidden state
        x = torch.relu(self.fc1(lstm_out))
        x = self.fc2(x)  # Apply sigmoid activation
        return x