import numpy as np
import pickle
import gzip
import os

def main(filename):
    outfile = filename.replace('dat.gz', 'npz')
    print('%s => %s' % (filename, outfile))

    if os.path.exists(os.path.join('data', outfile)):
        return

    data = pickle.load(gzip.open(os.path.join('pkl', filename)))

    labels = data[b'labels']

    data = data[b'data']
    data = np.asarray(data, dtype=np.float32)

    print(data, data.shape)

    np.savez_compressed(os.path.join('data', outfile), labels=labels, data=data)

def concat(filename):
    train = np.load(os.path.join('data', filename), allow_pickle=True)
    train_y, train_x = train['labels'], train['data']

    test = np.load(os.path.join('data', filename.replace('train', 'test')), allow_pickle=True)
    test_y, test_x = test['labels'], test['data']

    data = np.vstack([train_x, test_x])
    labels = np.vstack([train_y.reshape(train_y.shape[0], 1), test_y.reshape(test_y.shape[0], 1)])

    np.savez_compressed(os.path.join('data', filename.replace('train_', '')), labels=labels, data=data)

if __name__ == '__main__':
    for filename in os.listdir('pkl'):
        if filename.endswith('dat.gz'):
            main(filename)

    for filename in os.listdir('data'):
        if filename.startswith('train_'):
            concat(filename)

