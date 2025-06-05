import torch
from torch import nn

class AENet(nn.Module):
    def __init__(self, input_dim, block_size):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        # Encoder layers
        self.encoder = nn.Sequential(
            # First layer
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2),
            
            # Third layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2),
            
            # Fourth layer (latent space)
            nn.Linear(64, 16),
            nn.BatchNorm1d(16, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            # First layer
            nn.Linear(16, 64),
            nn.BatchNorm1d(64, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.Linear(64, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2),
            
            # Third layer
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2),
            
            # Fourth layer (output)
            nn.Linear(256, self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(z), z
