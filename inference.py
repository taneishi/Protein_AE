import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import argparse
import timeit
from model import AutoEncoder

def load_dataset(args, device):
    test = np.load(args.datafile, allow_pickle=True)['test']

    # create torch tensor from numpy array
    test = torch.FloatTensor(test).to(device)
    test = torch.utils.data.TensorDataset(test)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)

    return test_dataloader

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    test_dataloader = load_dataset(args, device)

    net = AutoEncoder(input_dim=1900, nlayers=args.nlayers, latent=100).to(device)
    net.load_state_dict(torch.load(args.modelfile))
    net.eval()

    # define loss function
    loss_func = nn.MSELoss(reduction='mean')

    test(test_dataloader, net, loss_func)

    test_loss = 0
    output = []

    for index, (data, ) in enumerate(test_dataloader, 1):
        with torch.no_grad():
            output.append(net(data))
        loss = loss_func(output[-1], data)
        test_loss += loss.item()

    print(' test_loss %6.3f' % (test_loss / index), end='')

    torch.save({
        'output': output,
        }, args.outputfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', required=True, type=str)
    parser.add_argument('--modelfile', required=True, type=str)
    parser.add_argument('--outputfile', required=True, type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--nlayers', default=4, type=int)
    args = parser.parse_args()

    main(args)
