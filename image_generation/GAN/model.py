import torch
import torch.nn as nn

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


def get_noise(n_samples, z_dim):
    # torch.randn() : 평균이 0이고 표준편차가 1인 가우시안 정규분포를 이용해 생성
    return torch.randn(n_samples, z_dim).to(device)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()

        def get_generator_block(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )

        self.gen = nn.Sequential(  # 10 -> 128 -> 256 -> 512 -> 1024 -> 784
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),  # hidden_dim * 2 = 256
            get_generator_block(hidden_dim * 2, hidden_dim * 4),  # hidden_dim * 4 = 512
            get_generator_block(hidden_dim * 4, hidden_dim * 8),  # hidden_dim * 8 = 1024
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()

        def get_discriminator_block(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.disc = nn.Sequential(  # 784 -> 512 -> 256 -> 128 -> 1
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc

# if __name__ == "__main__":
#     noise = get_noise(1000, 10)
#     print(noise.shape)

#     G = Generator()
#     image = G(noise)
#     print(image)

#     D = Discriminator()
#     prob = D(image)
#     print(prob)