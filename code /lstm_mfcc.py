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
from torch.utils.tensorboard import SummaryWriter   

from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

import librosa
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, batch_size, bi):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bi = 2 if bi == True else 1
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bi)
        
        self.fc = nn.Linear(self.bi*hidden_size, num_classes)
        # self.softmax = nn.Softmax(dim=1)



    def forward(self, x, device):
        h_0 = Variable(torch.zeros(
            self.num_layers * self.bi, self.batch_size, self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers* self.bi, self.batch_size, self.hidden_size))
        
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        # h_out = h_out.view(-1, self.hidden_size)
        h_out = ula[:, -1, :]
        # h_out = h_out.view(-1, self.hidden_size)


        
        out = self.fc(h_out)
        # out = self.softmax(out)
        
        return out
    
def train(lstm, device, data_loader_train, data_loader_dev, BATCH_SIZE):
    
    lstm.to(device)
    criterion.to(device)
    writer = SummaryWriter('summary_writer/')

    
    # Train the model
    for epoch in range(num_epochs):
        LOSS = 0
        batch = 1
        LOSS_list = []
        for x, y in tqdm(data_loader_train):
            x = x.to(device)
            y = y.to(device)

            outputs = lstm(x, device)

            # obtain the loss function
            loss = criterion(outputs, y[0])



            # print(outputs)
            # print('outputs:',outputs.shape)
            # print('y:',y.shape) 
            LOSS += loss
            LOSS_list.append(LOSS.item())
            batch += 1
            if batch == BATCH_SIZE:
                optimizer.zero_grad()

                LOSS.backward()

                optimizer.step()
                LOSS = 0
                batch = 1

            writer.add_scalar(save_path, LOSS, epoch)    


        dev_precision = test(lstm, data_loader_dev)

        print("Epoch: %d, loss: {%.3f}, dev_accuracy: {%.3f}" % (epoch, np.mean(LOSS_list), dev_precision))

        torch.save(lstm, 'ckp/{}.ckp'.format(save_path))
        
def compute_accuracy(output, target):
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
            correct_num, total_num = compute_accuracy(outputs, y)
            correct_list.append(correct_num)
            total_list.append(total_num)
        
        precision = sum(correct_list) / sum(total_list)
        
        return precision
    
if __name__ == '__main__':
    
    base_path = 'augmented_mfcc24_npy_files/'
    with open(base_path+'trainX.npy', 'rb') as f:
        trainX = np.load(f, allow_pickle=True)
    with open(base_path+'trainY.npy', 'rb') as f:
        trainY = np.load(f, allow_pickle=True)

    with open(base_path+'devX.npy', 'rb') as f:
        devX = np.load(f, allow_pickle=True)
    with open(base_path+'devY.npy', 'rb') as f:
        devY = np.load(f, allow_pickle=True)

    with open(base_path+'testX.npy', 'rb') as f:
        testX = np.load(f, allow_pickle=True)
    with open(base_path+'testY.npy', 'rb') as f:
        testY = np.load(f, allow_pickle=True)
        
    
    
    BATCH_SIZE = 8
    num_epochs = 50
    learning_rate = 0.01
    input_size = 24
    hidden_size = 16
    num_layers = 4
    num_classes = 41
    
    save_path = 'augmented_bi_{}layer_lstm-mfcc-{}epoch'.format(num_layers, num_epochs)
    
    data_train = MyDataset(trainX, trainY)
    data_loader_train = DataLoader(data_train, batch_size=1, shuffle=True, drop_last=True)
    data_loader_dev = DataLoader(MyDataset(devX, devY), batch_size=1, shuffle=True, drop_last=True)
    data_loader_test = DataLoader(MyDataset(testX, testY), batch_size=1, shuffle=True,  drop_last=True)
    
    # batch_train = iter(data_loader_train).next()
    # seq_len = batch_train[0].data.shape[0]

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, batch_size=1, bi=True)
    criterion = torch.nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training")
    train(lstm, device, data_loader_train, data_loader_dev, BATCH_SIZE)
    
    print('Testing')
    test_precision = test(lstm, data_loader_test)
    print('Test accuracy: {%.3f}', test_precision)