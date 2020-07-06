import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import train_test_split
import argparse
import timeit

from model import AutoEncoder

def load_dataset(datafile, batch_size, device):
    data = np.load(datafile, allow_pickle=True)['data']
    train_x, test_x = train_test_split(data, train_size=0.8, test_size=0.2)

    # create torch tensor from numpy array
    train_x = torch.FloatTensor(train_x).to(device)
    test_x = torch.FloatTensor(test_x).to(device)

    train = torch.utils.data.TensorDataset(train_x)
    test = torch.utils.data.TensorDataset(test_x)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train(dataloader, model, optimizer, loss_func, epoch):
    model.train()
    train_loss = 0

    for index, (data, ) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, data)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        print('\repoch %4d batch %4d/%4d train_loss %6.3f' % (epoch, index, len(dataloader), train_loss / index), end='')

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

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('Using %s device.' % device)

    train_dataloader, test_dataloader = load_dataset(args.datafile, args.batch_size, device)

    model = AutoEncoder(input_dim=1900, nlayers=5, latent=100).to(device)
    if args.modelfile:
        model.load_state_dict(torch.load(args.modelfile))

    print(model)

    # define our optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = nn.MSELoss()

    test_losses = []

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        train(train_dataloader, model, optimizer, loss_func, epoch)
        test_loss = test(test_dataloader, model, loss_func)

        print(' %5.2f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if test_loss <= min(test_losses):
            torch.save(model.state_dict(), 'model/%5.3f.pth' % min(test_losses))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='data/aponc_sda.npz')
    parser.add_argument('--modelfile', default=None, type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    print(vars(args))

    main(args)
