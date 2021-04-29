import torch
import torch.nn as nn
from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


class Generator(nn.Module):
    def __init__(self, d_noise=100, d_hidden=256):
        super().__init__()

        self.d_hidden = d_hidden

        self.generator = nn.Sequential(
            nn.Linear(d_noise, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 28 * 28),
            nn.Tanh()
        ).to(device)

    def forward(self, z):
        return self.generator(z)


class Discriminator(nn.Module):
    def __init__(self, d_hidden=256):
        super().__init__()

        self.d_hidden = d_hidden

        self.discriminator = nn.Sequential(
            nn.Linear(28 * 28, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.discriminator(x)


if __name__ == '__main__':
    # Generator 선언
    G = Generator()
    # 100차원의 Gaussian distribution에서 샘플링한 노이즈 z만들기
    z = torch.randn(100, device=device)
    fake_img = G(z)  # G(z)는 784차원(28x28)

    # Discriminator 선언
    D = Discriminator()
    res = D(fake_img)