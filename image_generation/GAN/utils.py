import os

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_tensor_images(image_name, image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    path = './res'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + '/' + image_name + '.png')