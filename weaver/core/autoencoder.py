#结构可参数化（输入维度、隐层、压缩维度）
#后续支持变体如 Denoising AE / VAE

import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=4, bottleneck_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
