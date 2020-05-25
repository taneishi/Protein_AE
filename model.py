import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.e1 = nn.Linear(1900, 1600)
        self.e2 = nn.Linear(1600, 1300)
        self.e3 = nn.Linear(1300, 1000)
        self.e4 = nn.Linear(1000, 700)
        self.e5 = nn.Linear(700, 400)
        # Latent View
        self.lv = nn.Linear(400, 100)
        # Decoder
        self.d1 = nn.Linear(100, 400)
        self.d2 = nn.Linear(400, 700)
        self.d3 = nn.Linear(700, 1000)
        self.d4 = nn.Linear(1000, 1300)
        self.d5 = nn.Linear(1300, 1600)
        self.output_layer = nn.Linear(1600, 1900)
        
    def forward(self, x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = F.relu(self.e3(x))
        x = F.relu(self.e4(x))
        x = F.relu(self.e5(x))
        x = torch.sigmoid(self.lv(x))
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        x = F.relu(self.d4(x))
        x = F.relu(self.d5(x))
        x = self.output_layer(x)
        return x
