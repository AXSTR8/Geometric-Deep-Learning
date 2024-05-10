import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from Models import CNN_1
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

#print(os.path.realpath(__file__))
s = os.path.dirname(os.path.realpath(__file__)) + "\Data"
#print(f"The directory is {s}")
print(type(s))

curr_dir = os.path.dirname(os.path.realpath(__file__))
print(curr_dir)
#model_dir = Path(r {s}+ "/Model_Parameters")

#print(model_dir)

train_data_dir = curr_dir + "\Data\Train"
train_data_path = Path(train_data_dir)
print(train_data_path)



# Define the data-transformer to convert the image data into PyTorch Tensors of a specified dimension

data_transformer = transforms.Compose([
    transforms.Resize(size=(256,256)), #Resize the image to 256x256
    transforms.ToTensor()
])

# Create the training data
train_data = datasets.ImageFolder(root=train_data_path, transform=data_transformer, target_transform=None)
