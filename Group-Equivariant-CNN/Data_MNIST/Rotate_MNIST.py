import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from wslpath import wslpath 
import tqdm
#from tqdm.auto import tqdm




curr_dir = os.path.dirname(os.path.realpath(__file__))
train_data_path = Path(curr_dir)
test_data_path = Path(curr_dir)

try:
    train_dataset = torchvision.datasets.MNIST(root=wslpath(train_data_path), train = True, transform=transforms.ToTensor(), download=False)
    test_dataset = torchvision.datasets.MNIST(root=wslpath(test_data_path), train = False, transform=transforms.ToTensor())
except:
    train_dataset = torchvision.datasets.MNIST(root=train_data_path, train = True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root=test_data_path, train = False, transform=transforms.ToTensor())

class MnistDataset(Dataset):
    def __init__(self, data_set, rotated: bool = True):

        images_list = [train_dataset[i][0] for i in range(len(train_dataset))]
        images =  torch.stack(list(images_list), dim=0).reshape(-1, 28, 28).numpy().astype(np.float32)
        labels_list = [torch.tensor(train_dataset[i][1]) for i in range(len(train_dataset))]
        #labels =  torch.stack(list(labels_list), dim=0).numpy().astype(np.float32)

        # images are padded to have shape 29x29.
        # this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
        pad = Pad((0, 0, 1, 1), fill=0)

        # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
        # we upsample an image by a factor of 3, rotate it and finally downsample it again
        resize1 = Resize(87) # to upsample
        resize2 = Resize(29) # to downsample

        totensor = ToTensor()

        if rotated:
            self.images = torch.empty((images.shape[0], 1, 29, 29))
            for i in tqdm(range(images.shape[0]), leave=False):
                img = images[i]
                img = Image.fromarray(img, mode='F')
                r = (np.random.rand() * 360.)
                self.images[i] = totensor(resize2(resize1(pad(img)).rotate(r, Image.BILINEAR))).reshape(1, 29, 29)
        else:
            self.images = torch.zeros((images.shape[0], 1, 29, 29))
            self.images[:, :, :28, :28] = torch.tensor(images).reshape(-1, 1, 28, 28)

        #labels_list = [train_dataset[i][1] for i in range(len(train_dataset))]
        self.labels = torch.stack(list(labels_list), dim=0).numpy().astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        return image, label

    def __len__(self):
        return len(self.labels)



train_dataset_rotated_path = train_data_path / "train_data_rotated.pt"
test_dataset_rotated_path = train_data_path / "test_data_rotated.pt"

if not Path(train_dataset_rotated_path).exists():
    train_dataset_rotated = MnistDataset(train_dataset, rotated=True)
    torch.save(train_dataset_rotated, train_dataset_rotated_path)

if not Path(test_dataset_rotated_path).exists():
    test_dataset_rotated = MnistDataset(test_dataset, rotated=True)
    torch.save(test_dataset_rotated, test_dataset_rotated_path)