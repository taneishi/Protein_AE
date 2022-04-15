import numpy as np
import torch
import pickle
import gzip
import os

def main(filename):
    outfile = filename.replace('dat.gz', 'npz')

    data = pickle.load(gzip.open('pkl/%s' % (filename)))

    labels = data[b'labels']

    data = data[b'data']
    data = np.asarray(data, dtype=np.float32)

    print(data, data.shape)

    np.savez_compressed('data/%s' % (outfile), labels=labels, data=data)

def concat(filename):
    train = np.load('data/%s' % (filename), allow_pickle=True)
    train_y, train_x = train['labels'], train['data']

    test = np.load('data/%s' % (filename.replace('train', 'test')), allow_pickle=True)
    test_y, test_x = test['labels'], test['data']

    data = np.vstack([train_x, test_x])
    labels = np.vstack([train_y.reshape(train_y.shape[0], 1), test_y.reshape(test_y.shape[0], 1)])

    np.savez_compressed('data/%s' % (filename.replace('train_', '')), labels=labels, data=data)

if __name__ == '__main__':
    for filename in os.listdir('pkl'):
        if filename.endswith('dat.gz'):
            main(filename)

    for filename in os.listdir('data'):
        if filename.startswith('train_'):
            concat(filename)
