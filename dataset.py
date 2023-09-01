import config
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle

class TCDataset(Dataset):
    def __init__(self, data, labels, id, word2idx, max_seq_length):
        self.data = data
        self.labels = labels
        self.id = id
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        label = label[1:-1].split(",")
        label = [float(i) for i in label]
        label = torch.tensor(label,dtype=torch.float32)
        id = self.id[idx]
        text_indices = [self.word2idx[word] for word in text if word in self.word2idx]
        if(len(text_indices)>=self.max_seq_length):
            text_indices = text_indices[:self.max_seq_length]  # Truncate if necessary
        else:
            a=len(text_indices)
            text_indices.extend([40000]*(self.max_seq_length-a))

        text_tensor = torch.tensor(text_indices)
        
        return text_tensor, label, id
    
if __name__ == "__main__":
    """
    taking a rain check
    """
    max_seq_length=config.MAX_SEQ_LENGTH
    data = pd.read_csv('./processed_data/p1_data.csv')
    data = data.dropna()

    X = data['text'].tolist()
    y = data['label'].tolist()
    # print(y)
    id = data['id'].tolist()

    with open(config.WORD2IDX_PATH, 'rb') as f:
        word2idx=pickle.load(f)
    dataset = TCDataset(X, y,id, word2idx, max_seq_length)
    loader = DataLoader(dataset=dataset, batch_size=32, num_workers=2, shuffle=True, pin_memory=True)

    for x, label, id in (loader):
        print(x.shape)
        print(x.dtype)
        print(label.shape)
        print(label.dtype)
        import sys
        sys.exit()
