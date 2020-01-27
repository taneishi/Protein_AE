#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

def show_torch_image(torch_tensor, name):
    plt.imshow(torch_tensor.numpy().reshape(28, 28), cmap='gray')
    plt.savefig('figure/%s.png' % name)

def load_dataset(batch_size):
    # Load dataset
    train = pd.read_csv('data/fashion-mnist_train.csv.gz')

    # normalization and preprocessing
    X = train.iloc[:,1:].values / 255.
    X = (X - 0.5) / 0.5

    Y = train.iloc[:,0].values

    print(X.shape, Y.shape)

    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.20, random_state=123)

    # create torch tensor from numpy array
    train_x_torch = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y_torch = torch.from_numpy(train_y)

    valid_x_torch = torch.from_numpy(valid_x).type(torch.FloatTensor)
    valid_y_torch = torch.from_numpy(valid_y)

    train = TensorDataset(train_x_torch, train_y_torch)
    valid = TensorDataset(valid_x_torch, valid_y_torch)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    show_torch_image(train_x_torch[1], 'train_sample')
    show_torch_image(valid_x_torch[1], 'valid_sample')

    return train_dataloader, valid_dataloader

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # encoder
        self.e1 = nn.Linear(28*28, 28)
        self.e2 = nn.Linear(28, 250)
        
        # Latent View
        self.lv = nn.Linear(250, 10)
        
        # Decoder
        self.d1 = nn.Linear(10, 250)
        self.d2 = nn.Linear(250, 500)
        
        self.output_layer = nn.Linear(500, 28*28)
        
    def forward(self,x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        
        x = torch.sigmoid(self.lv(x))
        
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        
        x = self.output_layer(x)
        return x

def train(dataloader, model, optimizer, loss_func, batch_size, device):
    model.train()
    losses = []
    EPOCHS = 10

    for epoch in range(EPOCHS):
        
        for index, (data, target) in enumerate(dataloader, 1):
            data = data.to(device)
        
            optimizer.zero_grad()
            pred = model(data)
            
            loss = loss_func(pred, data)
            losses.append(loss.cpu().data.item())
            
            # backpropagation
            loss.backward()
            optimizer.step()
            
            print('\repoch: %2d [%3d/%3d] loss: %5.3f' % (epoch, index, len(dataloader), loss.cpu().data.item()), end='')
        print('')

def test(dataloader, model, device):
    model.eval()
    predictions = []

    for data, target in dataloader:
            data = data.to(device)
            pred = model(data)
            
            for prediction in pred:
                predictions.append(prediction)
                   
    return predictions

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size = 100

    train_dataloader, valid_dataloader = load_dataset(batch_size)

    ae = AutoEncoder().to(device)
    print(ae)

    # define our optimizer and loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    train(train_dataloader, ae, optimizer, loss_func, batch_size, device)

    predictions = test(valid_dataloader, ae, device)
    print(len(predictions))

    show_torch_image(predictions[1].cpu().detach(), 'pred_sample')

if __name__ == '__main__':
    main()
