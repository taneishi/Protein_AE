import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import argparse
import timeit

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1900, nlayers=4, latent=100):
        super(AutoEncoder, self).__init__()
        delta = int((input_dim - latent) / (nlayers + 1))

        # Encoder
        encoder = []
        nunits = input_dim

        for layer in range(nlayers):
            encoder.append(nn.Linear(nunits, nunits - delta))
            nunits = nunits - delta
        self.encoder = nn.ModuleList(encoder)

        # Latent View
        self.lv = nn.Linear(nunits, latent)

        # Decoder
        decoder = []
        nunits = latent

        for layer in range(nlayers):
            decoder.append(nn.Linear(nunits, nunits + delta))
            nunits = nunits + delta
        self.decoder = nn.ModuleList(decoder)

        self.output_layer = nn.Linear(nunits, input_dim)
        
    def forward(self, x, activation=F.relu):
        for layer in self.encoder:
            x = activation(layer(x))

        x = torch.sigmoid(self.lv(x))

        for layer in self.decoder:
            x = activation(layer(x))

        x = self.output_layer(x)
        return x

def load_dataset(args, device):
    train_x = np.load(args.train_datafile, allow_pickle=True)['train']
    test_x = np.load(args.test_datafile, allow_pickle=True)['test']

    # create torch tensor from numpy array
    train_x = torch.FloatTensor(train_x).to(device)
    test_x = torch.FloatTensor(test_x).to(device)

    train = torch.utils.data.TensorDataset(train_x)
    test = torch.utils.data.TensorDataset(test_x)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train(dataloader, net, optimizer, loss_func, epoch):
    net.train()
    train_loss = 0

    for index, (data, ) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, data)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('epoch %4d batch %4d/%4d train_loss %6.3f' % (epoch, index, len(dataloader), train_loss / index), end='')

    return train_loss / index

def test(dataloader, net, loss_func):
    net.eval()
    test_loss = 0

    for index, (data, ) in enumerate(dataloader, 1):
        with torch.no_grad():
            output = net(data)
        loss = loss_func(output, data)
        test_loss += loss.item()

    print(' test_loss %6.3f' % (test_loss / index), end='')

    return test_loss / index

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    train_dataloader, test_dataloader = load_dataset(args, device)

    net = AutoEncoder(input_dim=1900, nlayers=args.nlayers, latent=100).to(device)

    # define our optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_func = nn.MSELoss(reduction='mean')

    for epoch in range(args.epochs):
        train(train_dataloader, net, optimizer, loss_func, epoch)
        test(test_dataloader, net, loss_func)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_datafile', required=True, type=str)
    parser.add_argument('--test_datafile', required=True, type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--nlayers', default=4, type=int)
    args = parser.parse_args()

    main(args)
