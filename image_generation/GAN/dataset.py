import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils


# from PIL import Image
# from glob import glob

def load_mnist():
    # grayscale 이미지에서 transform하기
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # RGB 이미지에서 transform하기
    # mnist_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # 데이터셋 다운로드 경로
    download_root = './'

    # 이미지 다운받기
    train_dataset = datasets.MNIST(download_root, train=True, transform=mnist_transform, download=True)
    test_dataset = datasets.MNIST(download_root, train=False, transform=mnist_transform, download=True)

    # Data loader를 사용해서 mnist datasets 로딩하기
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
    # Data가 잘 다운받아졌는지 출력으로 확인하기


#     for batch_idx, (x, target) in enumerate(train_loader):
#         if batch_idx % 10 == 0:
#             print(x.shape, target.shape)
#             print(len(train_loader.dataset))

# # 커스텀 데이터셋(Custom Dataset) 만들기
# class MNISTdataset(Dataset):
#     def __init__(self, img_path, transforms=None):
#         # 데이터셋의 전처리를 해주는 부분
#         self.img_path = img_path
#         self.transforms = transforms

#     def __len__(self):
#         # 데이터셋의 길이(샘플의 갯수)를 나타내는 부분
#         return len(glob(os.path.join(self.img_path, '*.jpg')))

#     def __getitem__(self, idx):
#         # 데이터셋에서 특정 1개의 샘플을 가져오는 함수


if __name__ == '__main__':
    # Custom dataset 만들어보고 싶은데, MNIST 데이터셋이라 그런지 잘 안된다...
    # 다음에 도전!
    load_mnist()