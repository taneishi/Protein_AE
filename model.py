import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1900):
        super(AutoEncoder, self).__init__()
        # encoder
        self.e1 = nn.Linear(input_dim, 1000)
        self.e2 = nn.Linear(1000, 500)
        # Latent View
        self.lv = nn.Linear(500, 100)
        # Decoder
        self.d1 = nn.Linear(100, 500)
        self.d2 = nn.Linear(500, 1000)
        self.output_layer = nn.Linear(1000, input_dim)
        
    def forward(self, x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = torch.sigmoid(self.lv(x))
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.output_layer(x)
        return x
