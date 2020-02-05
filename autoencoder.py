#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

def show_torch_image(torch_tensor, name):
    plt.imshow(torch_tensor.cpu().reshape(28, 28), cmap='gray')
    plt.savefig('figure/%s.png' % name)

def load_dataset(batch_size, device):
    # Load dataset
    train = np.load('data/fashion-mnist_train.npz', allow_pickle=True)['data']

    # normalization and preprocessing
    X = train[:,1:] / 255.
    X = (X - 0.5) / 0.5

    Y = train[:,0]

    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.20, random_state=123)

    print('train %d test %d' % (len(train_y), len(valid_y)))

    # create torch tensor from numpy array
    train_x_torch = torch.FloatTensor(train_x).to(device)
    train_y_torch = torch.ShortTensor(train_y).to(device)

    valid_x_torch = torch.FloatTensor(valid_x).to(device)
    valid_y_torch = torch.ShortTensor(valid_y).to(device)

    train = TensorDataset(train_x_torch, train_y_torch)
    valid = TensorDataset(valid_x_torch, valid_y_torch)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    valid_dataloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

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

def train(dataloader, model, optimizer, loss_func, batch_size):
    model.train()
    train_loss = 0
    EPOCHS = 10

    for epoch in range(EPOCHS):
        for index, (data, target) in enumerate(dataloader, 1):
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_func(pred, data)
            train_loss += loss.item()
            loss.backward() # backpropagation
            optimizer.step()
            print('\repoch: %2d [%3d/%3d] train_loss: %5.3f' % (epoch, index, len(dataloader), loss.item()), end='')
        print('')

def test(dataloader, model):
    model.eval()
    predictions = []

    for data, target in dataloader:
            pred = model(data)
            
            for prediction in pred:
                predictions.append(prediction)
                   
    return predictions

def main():
    batch_size = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    train_dataloader, valid_dataloader = load_dataset(batch_size, device)

    ae = AutoEncoder().to(device)
    print(ae)

    # define our optimizer and loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    train(train_dataloader, ae, optimizer, loss_func, batch_size)

    predictions = test(valid_dataloader, ae)
    print(len(predictions))

    show_torch_image(predictions[1].cpu().detach(), 'pred_sample')

if __name__ == '__main__':
    main()
