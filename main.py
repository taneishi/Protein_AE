import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import timeit
import sys
import os

def load_dataset(filename, batch_size, device):
    # Load dataset
    train = np.load(filename, allow_pickle=True)
    train_y, train_x = train['labels'], train['data']

    test = np.load(filename.replace('train', 'test'), allow_pickle=True)
    test_y, test_x = test['labels'], test['data']

    print('train', train_x.shape)
    print('test ', test_x.shape)
    assert train_x.shape[1] == test_x.shape[1]

    # create torch tensor from numpy array
    train_x_torch = torch.FloatTensor(train_x).to(device)
    test_x_torch = torch.FloatTensor(test_x).to(device)

    train = torch.utils.data.TensorDataset(train_x_torch)
    test = torch.utils.data.TensorDataset(test_x_torch)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.e1 = nn.Linear(1900, 1600)
        self.e2 = nn.Linear(1600, 1300)
        self.e3 = nn.Linear(1300, 1000)
        self.e4 = nn.Linear(1000, 700)
        self.e5 = nn.Linear(700, 400)
        # Latent View
        self.lv = nn.Linear(400, 100)
        # Decoder
        self.d1 = nn.Linear(100, 400)
        self.d2 = nn.Linear(400, 700)
        self.d3 = nn.Linear(700, 1000)
        self.d4 = nn.Linear(1000, 1300)
        self.d5 = nn.Linear(1300, 1600)
        self.output_layer = nn.Linear(1600, 1900)
        
    def forward(self,x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = F.relu(self.e3(x))
        x = F.relu(self.e4(x))
        x = F.relu(self.e5(x))
        x = torch.sigmoid(self.lv(x))
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        x = F.relu(self.d4(x))
        x = F.relu(self.d5(x))
        x = self.output_layer(x)
        return x

def train(dataloader, model, optimizer, loss_func):
    model.train()
    train_loss = 0

    for index, (data, ) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, data)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('\rbatch [%3d/%3d] train_loss %6.3f' % (index, len(dataloader), train_loss / index), end='')

    return train_loss / index

def test(dataloader, model, loss_func):
    model.eval()
    test_loss = 0

    for index, (data, ) in enumerate(dataloader, 1):
        with torch.no_grad():
            output = model(data)
        loss = loss_func(output, data)
        test_loss += loss.item()
    print(' test_loss %6.3f' % (test_loss / index), end='')

    return test_loss / index

def main():
    epochs = 1000
    batch_size = 100

    if len(sys.argv) < 2:
        sys.exit('no input filename')
    else:
        filename = sys.argv[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    train_dataloader, test_dataloader = load_dataset(filename, batch_size, device)

    model = AutoEncoder().to(device)
    print(model)

    # define our optimizer and loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses = [np.inf]
    test_losses = [np.inf]

    for epoch in range(epochs):
        epoch_start = timeit.default_timer()

        train_loss = train(train_dataloader, model, optimizer, loss_func)
        test_loss = test(test_dataloader, model, loss_func)

        print(' epoch [%4d/%4d] time %5.2f' % (epoch, epochs, (timeit.default_timer() - epoch_start)), end='')

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < min(test_losses[:-1]):
            torch.save(model.state_dict(), 'model.pth' % epoch)
            print(' model updated')
        else:
            print('')

        if min(test_losses) < min(test_losses[-10:]):
            break

if __name__ == '__main__':
    main()
