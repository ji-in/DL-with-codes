import argparse

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from model import get_noise
from utils import save_tensor_images


# checkpoint 저장 추가
# validation 추가 (best model 저장 추가)
# custom dataset 만들기
# evaluation 추가
# 서버 이상 있을 때, checkpoint부터 시작하는 코드 추가
# 파라미터 초기화 추가

class GAN(object):
    def __init__(self, args):
        super().__init__()

        self.n_epochs = args.n_epochs
        self.z_dim = args.z_dim
        self.display_step = args.display_step
        self.batch_size = args.batch_size
        self.lr = args.lr

        self.crit = nn.BCEWithLogitsLoss()

        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.is_cuda else 'cpu')

        self.dataloader = DataLoader(
            MNIST('.', download=False, transform=transforms.ToTensor()),
            batch_size=self.batch_size,
            shuffle=True)

        self.gen = Generator(self.z_dim).to(self.device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)

        self.disc = Discriminator().to(self.device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.lr)

    def get_disc_loss(self, real, num_images):
        fake_noise = get_noise(num_images, self.z_dim)  # 노이즈 생성
        fake = self.gen(fake_noise)  # 가짜 이미지 생성

        disc_fake_pred = self.disc(fake.detach())  # 왜 여기만 detach() 했지?
        disc_fake_loss = self.crit(disc_fake_pred, torch.zeros_like(disc_fake_pred))  # fake image의 label은 0

        disc_real_pred = self.disc(real)
        disc_real_loss = self.crit(disc_real_pred, torch.ones_like(disc_real_pred))  # real image의 label은 1

        disc_loss = (disc_fake_loss + disc_real_loss) / 2  # discriminator loss 는 disc_fake_loss 와 disc_real_loss 의 평균

        return disc_loss

    def get_gen_loss(self, num_images):
        fake_noise = get_noise(num_images, self.z_dim)  # 노이즈 생성
        fake = self.gen(fake_noise)  # 가짜 이미지 생성
        disc_fake_pred = self.disc(fake)
        gen_loss = self.crit(disc_fake_pred, torch.ones_like(disc_fake_pred))  # 생성한 이미지의 label은 1

        return gen_loss

    def train(self):
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        test_generator = True  # Whether the generator should be tested
        gen_loss = False

        for epoch in range(self.n_epochs):
            for real, _ in tqdm(self.dataloader):
                cur_batch_size = len(real)  # 현재 배치 사이즈 저장

                # real을 flatten 하게 만든다.
                real = real.view(cur_batch_size, -1).to(self.device)

                ### Update discriminator ###
                self.disc_opt.zero_grad()
                disc_loss = self.get_disc_loss(real, cur_batch_size)
                disc_loss.backward(retain_graph=True)  # retain_graph 알아보기
                self.disc_opt.step()

                # For testing purposes, to keep track of the generator weights
                if test_generator:
                    old_generator_weights = self.gen.gen[0][0].weight.detach().clone()
                    # generator의 가장 처음 weights 를 저장한다.

                ### Update generator ###
                self.gen_opt.zero_grad()
                gen_loss = self.get_gen_loss(cur_batch_size)
                gen_loss.backward()
                self.gen_opt.step()

                # For testing purposes, to check that your code changes the generator weights
                if test_generator:
                    assert torch.any(self.gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                    # torch.any(input) : input에는 Tensor가 들어간다. input 안의 수(or bool) 중에서 True가 하나 이상 있으면 tensor(True)를 반환한다.
                    # 나의 코드가 generator weights를 변화시켰는지 확인한다.

                # 딕셔너리는 items() 함수를 사용하면 딕셔너리에 있는 키와 값들의 쌍을 얻을 수 있다.
                # >>> car = {"name": "BMW", "price": "8000"}
                # >>> car.items()
                # >>> dict_items([('name', 'BMW'), ('price', '7000')])

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_loss.item() / self.display_step

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / self.display_step

                ### Visualization code ###
                if cur_step % self.display_step == 0 and cur_step > 0:
                    print(
                        f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")

                    fake_noise = get_noise(cur_batch_size, self.z_dim)
                    fake = self.gen(fake_noise)
                    save_tensor_images("fake_" + str(cur_step), fake)
                    save_tensor_images("real_" + str(cur_step), real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                cur_step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=64, help='a dimension of input noise')
    parser.add_argument('--display_step', type=int, default=5, help='size of results is display_step * display_step')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    args = parser.parse_args()

    model = GAN(args)
    model.train()