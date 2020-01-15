# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="967a53c6595bcd7d3d61584b69c278715afbf504"
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')

# %%
batch_size = 100


# %% _uuid="da094aee0f9612e9297005def1ccaf3da871d86a"
def show_torch_image(torch_tensor):
    plt.imshow(torch_tensor.numpy().reshape(28, 28), cmap='gray')
    plt.show()


# %% _uuid="b732b6d19c25a1a0552a77614b5a86a34a4a163b"
#Load dataset
train = pd.read_csv("fashion-mnist_train.csv")

#normalization and preprocessing
X = train.iloc[:,1:].values / 255.
X = (X-0.5) / 0.5

Y = train.iloc[:,0].values

print(X.shape, Y.shape)

# %% _uuid="2267f9736dd45e2bbe993a60efba11ca9eb4e0c7"
trn_x, val_x, trn_y, val_y = train_test_split(X, Y, test_size=0.20, random_state=123)

# %% _uuid="157bd76d6f031d43843cc533a10df7cff60ea852"
#create torch tensor from numpy array
trn_x_torch = torch.from_numpy(trn_x).type(torch.FloatTensor)
trn_y_torch = torch.from_numpy(trn_y)

val_x_torch = torch.from_numpy(val_x).type(torch.FloatTensor)
val_y_torch = torch.from_numpy(val_y)

trn = TensorDataset(trn_x_torch, trn_y_torch)
val = TensorDataset(val_x_torch, val_y_torch)

trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# %% _uuid="0659e9474ffd51685287b5eb586f5fde9699b3ae"
show_torch_image(trn_x_torch[1])


# %% _uuid="e8e02729a5d6b6b7c637fa25a316784d65313375"
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


# %% _uuid="b554e18660e2e1c308e5ee4e1d10eef08a7a5361"
ae = AutoEncoder().to(device)
print(ae)

# %% _uuid="e7fa50b1fb713cfe81ea23709196aed7aa5b2d86"
# define our optimizer and loss function
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

# %% _uuid="aceddc5bf85f398528d9a7c7bcc63339b35342e7"
losses = []
EPOCHS = 10
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(trn_dataloader):
        data = data.to(device)
    
        optimizer.zero_grad()
        pred = ae(data)
        
        loss = loss_func(pred, data)
        losses.append(loss.cpu().data.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Display
        if batch_idx % 10 == 1:
            print('\rTrain Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1,
                EPOCHS,
                batch_idx * len(data), 
                len(trn_dataloader.dataset),
                batch_size * batch_idx / len(trn_dataloader), 
                loss.cpu().data.item()), 
                end='')
    print('')

# %% _uuid="6d7f0566578b9c9ba086b7456c5ee1fca08af22b"
ae.eval()
predictions = []

for data, target in val_dataloader:
        data = data.to(device)
        pred = ae(data)
        
        for prediction in pred:
            predictions.append(prediction)
               
len(predictions)

# %% _uuid="96f44ec49ed23faa1e34cbfdfd72bb9e1c1afe7a"
show_torch_image(val_x_torch[1])

# %% _uuid="5c723e8d1d9ccbe2693f862ac7b94b16b1c7ab83"
show_torch_image(predictions[1].to('cpu').detach())

# %% _uuid="9ef591347d96974556203a89c0bb2ddc4f68c5e6"
