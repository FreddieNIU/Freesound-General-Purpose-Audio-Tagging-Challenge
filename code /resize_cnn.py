import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d,BatchNorm1d, Dropout
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter   
import torchvision.models as tvmodels

from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import cv2

import librosa
import os



def batch_resize(maximum, feature_size, trainX):
    padded_trainX = [0 for i in range(len(trainX))]
    for i in range(len(trainX)):
        a = cv2.resize(np.array(trainX[i]), (maximum, feature_size)).T
        padded_trainX[i] = torch.stack([torch.tensor(a) for j in range(3)])
    return np.array(padded_trainX)

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
    
class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()

        # self.cnn_layers = Sequential(
        #     # define a 2D convolutional layer
        #     Conv2d(1, 3, kernel_size=(5, 5), stride=1, padding=(0,0)),
        #     ReLU(inplace=True),
            # BatchNorm2d(32),
            # MaxPool2d(kernel_size=2, stride=2),
            
#             Conv2d(64, 32, kernel_size=(10, 10), stride=1, padding=(0,0)),
#             ReLU(inplace=True),

#             # Defining another 2D convolution layer
#             Conv2d(32, 16, kernel_size=(3, 3), stride=1),
#             ReLU(inplace=True),
#             # BatchNorm2d(16),
#             # MaxPool2d(kernel_size=2, stride=2),

#             # Defining another 2D convolution layer
#             Conv2d(16, 8, kernel_size=(3, 3), stride=1),
#             ReLU(inplace=True),
#             BatchNorm2d(8),
        # )
        
        self.vgg16 = tvmodels.vgg16(pretrained=True)

        self.linear_layers = Sequential(
            
            
            # Linear(3839840, 10000),
            # Linear(10000, 1000),
            
            Linear(1000,256),
            BatchNorm1d(256),
            ReLU(inplace=True),
            # Dropout(0.2),
            
            Linear(256, 128),
            BatchNorm1d(128),
            ReLU(inplace=True),
            # Dropout(0.1),
            
            Linear(128, 41),
            Softmax(dim=1)
        )
        
        
    # Defining the forward pass
    def forward(self, x):
        # x = self.cnn_layers(x)
        x = self.vgg16(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        # print(x.shape)
        # print(torch.max(x, dim=1))
        return x
    
def compute_accuracy(output, target):
    target_hat = torch.max(output, dim=1).indices
    correct_num = int(sum(target_hat==target))
    total_num = target.shape[0]
    # print(target_hat, target, correct_num, total_num)
    return correct_num, total_num


def test(model, data_loader):
    model.eval()
    # accuracy
    with torch.no_grad():
        correct_list = []
        total_list = []
        for x, y in tqdm(data_loader):
            x = x.to(device)
            y = y.reshape(-1,).long().to(device)
            
            outputs = model(x)
            correct_num, total_num = compute_accuracy(outputs, y)
            correct_list.append(correct_num)
            total_list.append(total_num)
        
        accuracy = sum(correct_list) / sum(total_list)
        
 
    return accuracy

def train_net(model, train_data_iter,val_data_iter, epochs, save_path):
    model.train()
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        train_loss = []
        i=0
        for x,y in tqdm(train_data_iter):
            x = x.float().to(device)
            y = y.reshape(-1,).long().to(device)
            #x = x.float()
            #y = y.float() 
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            train_loss.append(loss.item())
            # print(out, y)
            # print(out.shape, y.shape)
            
            loss.backward()
            optimizer.step()
            
            
        dev_accuracy = test(model,val_data_iter) 
        
        # val_loss = []
        # for x,y in tqdm(val_data_iter):
        #     x= x.float().to(device)
        #     y = y.float().to(device)
        #     # x, y = data[0].to(device), data[1].to(device)
        #     # x = x.float()
        #     # y = y.float()
        #     out = model(x)
        #     loss_val = criterion(out, y)
        #     val_loss.append(loss_val.item())
        # print("train loss:", np.mean(train_loss), 'dev loss: ', np.mean(val_loss))

        print("train loss:", np.mean(train_loss), 'dev accuracy: ', dev_accuracy)
        
        torch.save(model, 'ckp/{}.ckp'.format(save_path))
        
    print("finish train")
    
if __name__ == '__main__':

    base_path = 'augmented_stft_npy_files//'
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
        
    padded_trainX = batch_resize(445, 241, trainX)
    padded_devX = batch_resize(445, 241, devX)
    padded_testX = batch_resize(445, 241, testX)
    
    BATCH_SIZE = 8
    
    data_loader_train = DataLoader(MyDataset(padded_trainX, trainY), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    data_loader_dev = DataLoader(MyDataset(padded_devX, devY), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    data_loader_test = DataLoader(MyDataset(padded_testX, testY), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = 50
    learning_rate = 0.001
    save_path = 'fine_tune_resize_vgg16_{}epoch_{}lr'.format(epochs, learning_rate)

    # model = CNN()
    model = torch.load('ckp/lp_resize_cnn_3conv_3linear_20epoch_0.001lr_02.ckp')
    # defining the optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    model.to(device)
    criterion.to(device)
    print(model)


    train_net(model, data_loader_train, data_loader_dev, epochs=epochs, save_path=save_path)

