import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split
import argparse
import timeit
import os

from model import AutoEncoder

def load_dataset(args, device, world_size):
    data = np.load(args.datafile, allow_pickle=True)['data']
    train_x, test_x = train_test_split(data, train_size=0.8, test_size=0.2)

    # create torch tensor from numpy array
    train_x = torch.FloatTensor(train_x).to(device)
    test_x = torch.FloatTensor(test_x).to(device)

    train = torch.utils.data.TensorDataset(train_x)
    test = torch.utils.data.TensorDataset(test_x)

    if world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=sampler_train)
        sampler_test = torch.utils.data.distributed.DistributedSampler(test)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, sampler=sampler_test)
    else:
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
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('Using %s device.' % device)

    world_size = int(os.environ[args.env_size]) if args.env_size in os.environ else 1
    local_rank = int(os.environ[args.env_rank]) if args.env_rank in os.environ else 0

    if local_rank == 0:
        print(vars(args))

    if world_size > 1:
        print('rank: {}/{}'.format(local_rank+1, world_size))
        torch.distributed.init_process_group(
                backend='gloo',
                init_method='file://%s' % args.tmpname,
                rank=local_rank,
                world_size=world_size)

    train_dataloader, test_dataloader = load_dataset(args, device, world_size)

    net = AutoEncoder(input_dim=1900, nlayers=args.nlayers, latent=100).to(device)

    if world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net)

    if args.modelfile:
        net.load_state_dict(torch.load(args.modelfile))

    # define our optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = nn.MSELoss(reduction='mean')

    test_losses = []

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        train(train_dataloader, net, optimizer, loss_func, epoch)
        test_loss = test(test_dataloader, net, loss_func)

        print(' %5.2f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if test_loss <= min(test_losses):
            torch.save(net.state_dict(), 'model/%5.3f.pth' % min(test_losses))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', required=True, type=str)
    parser.add_argument('--modelfile', default=None, type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--nlayers', default=4, type=int)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--tmpname', default='tmpfile', type=str)
    parser.add_argument('--env_size', default='WORLD_SIZE', type=str)
    parser.add_argument('--env_rank', default='RANK', type=str)
    args = parser.parse_args()

    main(args)
