import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1900, nlayers=4, latent=100):
        super(AutoEncoder, self).__init__()
        delta = int((input_dim - latent) / (nlayers + 1))
        # Encoder
        encoder = []
        nunits = input_dim

        for layer in range(nlayers):
            encoder.append(nn.Linear(nunits, nunits - delta))
            nunits = nunits - delta
        self.encoder = nn.ModuleList(encoder)

        # Latent View
        self.lv = nn.Linear(nunits, latent)

        # Decoder
        decoder = []
        nunits = latent

        for layer in range(nlayers):
            decoder.append(nn.Linear(nunits, nunits + delta))
            nunits = nunits + delta
        self.decoder = nn.ModuleList(decoder)

        self.output_layer = nn.Linear(nunits, input_dim)
        
    def forward(self, x, activation=F.relu):
        for layer in self.encoder:
            x = activation(layer(x))

        x = torch.sigmoid(self.lv(x))

        for layer in self.decoder:
            x = activation(layer(x))

        x = self.output_layer(x)
        return x
