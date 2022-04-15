import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import timeit

from model import AutoEncoder

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    data = np.load(args.datafile)['data']
    # labels for each amino acid pairs, not used.
    labels = np.load(args.datafile)['labels']

    train_data = torch.FloatTensor(data).to(device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    net = AutoEncoder(input_dim=data.shape[1], num_layers=args.num_layers, latent=100)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        net.train()
        start_time = timeit.default_timer()
        train_loss = 0
        for index, data in enumerate(train_loader, 1):
            output = net(data)
            loss = criterion(output, data)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('\repoch %3d/%3d batch %3d/%3d' % (epoch, args.epochs, index, len(train_loader)), end='')
            print(' train_loss %6.3f' % (train_loss / index), end='')

        print(' %4.1fsec' % (timeit.default_timer() - start_time))

    torch.save(net.state_dict(), args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='data/pdzapo_sda.npz', type=str)
    parser.add_argument('--model_path', default='model/pdzapo.pth', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
