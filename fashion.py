import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
import timeit
import os

def load_dataset(batch_size, device):
    # Load dataset
    train = np.load('data/fashion-mnist_train.npz', allow_pickle=True)['data']
    test = np.load('data/fashion-mnist_test.npz', allow_pickle=True)['data']

    # normalization and preprocessing
    train_x = train[:,1:] / 255.
    train_x = (train_x - 0.5) / 0.5
    train_y = train[:,0]

    test_x = test[:,1:] / 255.
    test_x = (test_x - 0.5) / 0.5
    test_y = test[:,0]

    print('train %d test %d' % (len(train_y), len(test_y)))

    # create torch tensor from numpy array
    train_x_torch = torch.FloatTensor(train_x).to(device)
    train_y_torch = torch.ShortTensor(train_y).to(device)

    test_x_torch = torch.FloatTensor(test_x).to(device)
    test_y_torch = torch.ShortTensor(test_y).to(device)

    train = torch.utils.data.TensorDataset(train_x_torch, train_y_torch)
    test = torch.utils.data.TensorDataset(test_x_torch, test_y_torch)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

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

def show_torch_image(tensor, name):
    plt.imshow(tensor.reshape(28, 28), cmap='gray')
    plt.savefig('figure/%s.png' % name)

def plot():
    # Load dataset
    train = np.load('data/fashion-mnist_train.npz', allow_pickle=True)['data']
    test = np.load('data/fashion-mnist_test.npz', allow_pickle=True)['data']

    # normalization and preprocessing
    train_x = train[:,1:] / 255.
    train_x = (train_x - 0.5) / 0.5
    train_y = train[:,0]

    test_x = test[:,1:] / 255.
    test_x = (test_x - 0.5) / 0.5
    test_y = test[:,0]

    show_torch_image(train_x[1], 'train_sample')
    show_torch_image(test_x[1], 'test_sample')

    if os.path.exists('prediction.npy'): 
        prediction = np.load('prediction.npy', allow_pickle=True)
        show_torch_image(predictions[1].cpu().detach(), 'pred_sample')

def train(dataloader, model, optimizer, loss_func, batch_size):
    model.train()
    train_loss = 0
    EPOCHS = 10

    for epoch in range(EPOCHS):
        epoch_start = timeit.default_timer()
        for index, (data, target) in enumerate(dataloader, 1):
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_func(pred, data)
            train_loss += loss.item()
            loss.backward() # backpropagation
            optimizer.step()
            print('\repoch %2d [%3d/%3d] train_loss %5.3f' % (epoch, index, len(dataloader), loss.item()), end='')
        print(' time %5.2f' % (timeit.default_timer() - epoch_start))

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

    train_dataloader, test_dataloader = load_dataset(batch_size, device)

    ae = AutoEncoder().to(device)
    print(ae)

    # define our optimizer and loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    train(train_dataloader, ae, optimizer, loss_func, batch_size)

    predictions = test(test_dataloader, ae)
    print(len(predictions))

if __name__ == '__main__':
    main()
    plot()
