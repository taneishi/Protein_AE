import numpy as np
import pickle
import gzip
import os

def main(filename):
    data = pickle.load(gzip.open(os.path.join('data', filename)))

    labels = data[b'labels']

    data = data[b'data']
    data = np.asarray(data, dtype=np.float32)

    print(data, data.shape)

    outfile = os.path.join('data', filename[:-6]+'npz')

    np.savez_compressed(outfile, labels=labels, data=data)

if __name__ == '__main__':
    for filename in os.listdir('data'):
        if filename.endswith('dat.gz'):
            main(filename)
