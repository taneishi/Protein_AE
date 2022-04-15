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
    print('input shape is', data.shape)

    test_data = torch.FloatTensor(data).to(device)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    net = AutoEncoder(input_dim=data.shape[1], num_layers=args.num_layers, latent=100)
    net = net.to(device)
    net.load_state_dict(torch.load(args.model_path))

    criterion = nn.MSELoss()

    net.eval()
    start_time = timeit.default_timer()
    output = [] 
    test_loss = 0
    for index, data in enumerate(test_loader, 1):
        with torch.no_grad():
            outputs = net(data)

        loss = criterion(outputs, data)
        test_loss += loss.item()

        output.append(outputs.detach().cpu().numpy())

        print('\rtest_loss %6.3f' % (test_loss / index), end='')

    print(' %4.1fsec' % (timeit.default_timer() - start_time))

    output = np.vstack(output)
    print('output shape is', output.shape)
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='data/pdzapo_sda.npz', type=str)
    parser.add_argument('--model_path', default='model/pdzapo.pth', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
