# Autoencoder-based Detection of Protein Dynamic Allostery 

The paper *"Autoencoder-based detection of the residues involved in G protein-coupled receptor signaling"* referring to this implementation has been published in [Scientific Reports](https://doi.org/10.1038/s41598-021-99019-z).

## Usage

```bash
bash run.sh
```

## Results

```bash
python train.py

epoch   1/ 30 batch 438/438 train_loss 41.496  2.4sec
epoch   2/ 30 batch 438/438 train_loss  1.273  2.4sec
epoch   3/ 30 batch 438/438 train_loss  1.018  2.4sec
epoch   4/ 30 batch 438/438 train_loss  1.014  2.4sec
epoch   5/ 30 batch 438/438 train_loss  1.041  2.4sec
...
epoch  26/ 30 batch 438/438 train_loss  0.999  2.4sec
epoch  27/ 30 batch 438/438 train_loss  1.049  2.4sec
epoch  28/ 30 batch 438/438 train_loss  0.982  2.4sec
epoch  29/ 30 batch 438/438 train_loss  1.010  2.4sec
epoch  30/ 30 batch 438/438 train_loss  0.989  2.4sec
```

```bash
python infer.py

input shape is (4371, 1500)
test_loss  0.985  0.2sec
output shape is (4371, 1500)
[[14.519997  14.478731  14.310144  ... 14.534247  14.858824  14.858355 ]
 [18.514946  18.46142   18.226017  ... 18.485933  18.904604  18.89699  ]
 [12.024682  11.992374  11.863634  ... 12.064727  12.330826  12.334305 ]
 ...
 [ 5.9411445  5.934773   5.89766   ...  6.036261   6.154844   6.1705303]
 [18.804516  18.750196  18.509752  ... 18.772165  19.19775   19.189568 ]
 [20.987549  20.927832  20.647852  ... 20.928368  21.406742  21.393595 ]]
```
