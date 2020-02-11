#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import timeit
import os

def show_torch_image(tensor, name):
    plt.imshow(tensor.reshape(28, 28), cmap='gray')
    plt.savefig('figure/%s.png' % name)

def main():
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

if __name__ == '__main__':
    main()
