import argparse

import torch
from torchvision import transforms
import torch.nn as nn

import numpy as np

from models import Generator, Discriminator
from dataset import load_mnist

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


class GAN(object):
    def __init__(self, args, generator=Generator, discriminator=Discriminator):
        super().__init__()

        # 환경 설정하기
        self.args = args
        self.gpu_id = args.gpu_id
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs

        # MNIST 데이터셋 불러오기
        self.train_loader, self.test_loader = load_mnist()

        # Generator와 Discriminator 초기화하기
        self.G = generator()
        self.D = discriminator()

        # 로스 함수 - Binary Cross Entropy 사용
        self.criterion = nn.BCELoss()

        # Optimizer
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002)

    def _train(self):
        self.G.train()
        self.D.train()

        # train_loader에서 batch크기 만큼 img_batch, label_batch 불러오기
        for img_batch, label_batch in self.train_loader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            # ====================================================== #
            # discriminator training 하기
            # ====================================================== #

            # 100차원의 Gaussian distribution에서 샘플링한 노이즈 z만들기
            z = torch.randn(100).to(device)

            # discriminator optimizer 초기화
            self.d_optimizer.zero_grad()

            ### real image를 사용해서 BCE Loss 계산하기

            # 실제 이미지 D에 넣기
            p_real = self.D(img_batch.view(-1, 28 * 28))
            real_labels = torch.ones_like(p_real).to(device)
            d_loss_real = self.criterion(p_real, real_labels)

            ### fake image를 사용해서 BCE Loss 계산하기

            # 랜덤 노이즈 D에 넣기
            p_fake = self.D(self.G(z))
            fake_labels = torch.zeros_like(p_fake).to(device)
            d_loss_fake = self.criterion(p_fake, fake_labels)

            ### discriminator의 최종 loss
            d_loss = d_loss_real + d_loss_fake

            # 모멘텀 적용하고, 파라미터 업데이트하기
            d_loss.backward()
            self.d_optimizer.step()

            # ====================================================== #
            # generator training 하기
            # ====================================================== #

            # generator optimizer 초기화
            self.g_optimizer.zero_grad()

            # 100차원의 Gaussian distribution에서 샘플링한 노이즈 z만들기
            z = torch.randn(100).to(device)
            # fake image 만들기
            p_fake = self.D(self.G(z))
            real_labels = torch.ones_like(p_fake).to(device)
            g_loss = self.criterion(p_fake, real_labels)

            # 모멘텀 적용하고, 파라미터 업데이트 하기
            g_loss.backward()
            self.g_optimizer.step()

    def train(self):
        i = 0
        for epoch in range(self.num_epochs):
            i += 1
            self._train()
            print(str(i) + '번 째 epoch')

        torch.save({
            'epoch': epoch,
            'g_state_dict': self.G.state_dict(),
            'd_state_dict': self.D.state_dict(),
        }, './checkpoint.pt')

    def test(self):
        from matplotlib import pyplot as plt

        with torch.no_grad():
            checkpoint = torch.load('./checkpoint.pt')

            G = Generator()
            G.load_state_dict(checkpoint['g_state_dict'])
            G.eval()

            z = torch.randn(100).to(device)
            img = G(z).view(28, 28, -1)
            plt.imshow(img.numpy(), cmap='gray')
            plt.savefig('savefig.png')


#     def evaluate(self):

#         p_real, p_fake = 0., 0.

#         self.G.eval()
#         self.D.eval()

#         for img_batch, label_batch in self.test_loader:
#             img_batch, label_batch = img_batch.to(device), label_batch.to(device)

#             with torch.autograd.no_grad():
#                 p_real += (torch.sum(discriminator(img_batch.view(-1, 28*28))).item())/10000.
#             p_fake += (torch.sum(discriminator(generator(sample_z(batch_size, d_noise)))).item())/10000.


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model_fn', required=True)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])

    args = parser.parse_args()
    # print(args)

    gan = GAN(args)

    if args.mode == 'train':
        gan.train()
    else:
        gan.test()


if __name__ == '__main__':
    main()