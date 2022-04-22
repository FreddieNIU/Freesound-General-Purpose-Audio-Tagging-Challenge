import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

import librosa
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

BATCH_SIZE = 64

class MyDataset(Dataset):
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
    
    def __getitem__(self, idx):
        # return torch.FloatTensor(self.trainX[idx]), torch.FloatTensor(self.trainY[idx])
        return self.trainX[idx], self.trainY[idx]
        
    def __len__(self):
        assert np.array(self.trainX).shape[0] == np.array(self.trainY).shape[0]
        return np.array(self.trainX).shape[0]

def collate_fn(data):
    
    sequence, label = [], []
    for (_sequence, _label) in data:
        sequence.append(_sequence)
        label.append(int(_label))
    sequence.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in sequence]
    sequence = pad_sequence(sequence, batch_first=True, padding_value=0)
    sequence = pack_padded_sequence(sequence, seq_len, batch_first=True)

    return sequence,torch.tensor(np.array(label))

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, batch_size):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, device):
        h_0 = Variable(torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size))
        
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        # h_out = h_out.view(-1, self.hidden_size)
        h_out = h_out[-1, :, :]
        
        out = self.fc(h_out)
        
        return out
    
def train(lstm, device, data_loader_train, data_loader_dev):
    
    lstm.to(device)
    criterion.to(device)

    # Train the model
    for epoch in range(num_epochs):
        for x, y in tqdm(data_loader_train):
            x = x.to(device)
            y = y.to(device)

            outputs = lstm(x, device)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, y)
            # print('outputs:',outputs.shape)
            # print('y:',y.shape) 

            loss.backward()

            optimizer.step()

        dev_precision = test(lstm, data_loader_dev)

        print("Epoch: %d, loss: %1.5f, dev_precision: %2f" % (epoch, loss.item(), dev_precision))
        
def compute_precision(output, target):
    target_hat = torch.max(output, dim=1).indices
    correct_num = int(sum(target_hat==target))
    total_num = target.shape[0]
    return correct_num, total_num

def test(model, data_loader):
    with torch.no_grad():
        correct_list = []
        total_list = []
        for x, y in tqdm(data_loader):
            x = x.to(device)
            y = y.to(device)

            outputs = lstm(x, device)
            correct_num, total_num = compute_precision(outputs, y)
            correct_list.append(correct_num)
            total_list.append(total_num)
        
        precision = sum(correct_list) / sum(total_list)
        
        return precision
    
if __name__ == '__main__':
    
    with open('npy_files/trainX.npy', 'rb') as f:
        trainX = np.load(f, allow_pickle=True)
    with open('npy_files/trainY.npy', 'rb') as f:
        trainY = np.load(f, allow_pickle=True)

    with open('npy_files/devX.npy', 'rb') as f:
        devX = np.load(f, allow_pickle=True)
    with open('npy_files/devY.npy', 'rb') as f:
        devY = np.load(f, allow_pickle=True)

    with open('npy_files/testX.npy', 'rb') as f:
        testX = np.load(f, allow_pickle=True)
    with open('npy_files/testY.npy', 'rb') as f:
        testY = np.load(f, allow_pickle=True)
        
    data_train = MyDataset(trainX, trainY)
    data_loader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    data_loader_dev = DataLoader(MyDataset(devX, devY), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    data_loader_test = DataLoader(MyDataset(testX, testY), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    batch_train = iter(data_loader_train).next()
    
    num_epochs = 20
    learning_rate = 0.001
    input_size = 1
    hidden_size = 16
    num_layers = 1
    num_classes = 41
    seq_len = batch_train[0].data.shape[0]

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length=seq_len, batch_size=BATCH_SIZE)
    criterion = torch.nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training")
    train(lstm, device, data_loader_train, data_loader_dev)
    
    print('Testing')
    test_precision = test(lstm, data_loader_test)
    print('Test precision: %2f', test_precision)