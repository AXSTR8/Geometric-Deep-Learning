import os
import sys
import torch
import escnn
#import torch.nn as nn
from escnn import gspaces
from escnn import nn as nn
import Models
from Rotate_MNIST import MnistDataset
import Models_MNIST
from Models_MNIST import  CNN, DR_Equivariant_CNN, SO2SteerableCNN
import numpy as np
from PIL import Image
from pathlib import Path
from wslpath import wslpath
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

# Define the GPU as standard device if it is possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the current directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

# Specify the directory for the training and test data
train_data_directory = curr_dir + "/Data-MNIST/train_data_rotated.pt"
test_data_directory = curr_dir + "/Data-MNIST/test_data_rotated.pt"

# Load the training and test data
try: 
    train_dataset_rotated = torch.load(wslpath(train_data_directory))
    test_dataset_rotated = torch.load(wslpath(test_data_directory))
except:
    train_dataset_rotated = torch.load(Path(train_data_directory))
    test_dataset_rotated = torch.load(Path(test_data_directory))

# Define a dataloader for the training and test datasets
train_dataloader = DataLoader(dataset=train_dataset_rotated, batch_size=32,shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset_rotated, batch_size=32,shuffle=False)

# Define the training loop for a model
def train(model: torch.nn.Module,epochs=10, lr=1e-4, dataloader = train_dataloader, model_path = None, model_loss_path = None, model_accuracy_path = None):

    # Define the optimizer and the loss function and create a loss tracking list
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_tracker = []
    test_accuracy_tracker = []

    for epoch in range(epochs):
        model.train()
        for i, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            # Push the images and targets to the device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_tracker.append(loss.cpu().detach().numpy())
            del images, outputs, targets, loss

        test_accuracy = test(model, test_dataloader)
        test_accuracy_tracker.append(test_accuracy)
        print(f"{str(model).split('(')[0]}: epoch {epoch+1} and the test accuracy is {test_accuracy} ")

        # Save the parameters of the trained models if a path was provided
        if model_path != None:
            torch.save(model.state_dict(), model_path)
        
        # Save the losses of the trained models
        if model_loss_path != None:
            with open(Path(model_loss_path) , "w") as f:
                for item in loss_tracker:
                    f.write("%s\n" % item)
    
        # Save the losses of the trained models
        if model_accuracy_path != None:
            with open(Path(model_accuracy_path), "w") as f:
                for item in test_accuracy_tracker:
                    f.write("%s\n" % item)

# Define the test loop for a model
def test(model: torch.nn.Module, dataloader):
    # test over the full rotated test set
    total = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)

            _, prediction = torch.max(output.data, 1)
            total += targets.shape[0]
            correct += (prediction == targets).sum().item()
    return correct/total*100.

# Define the models
# Conventional CNN
CNN_Model = CNN(10).to(device)

# Create the C16 Equivaraint and a SO(2) Steerable CNN
GECNN_Model_1 = DR_Equivariant_CNN(10).to(device)
GECNN_Model_2 = SO2SteerableCNN(10).to(device)

# Combine the models in a list
Models = [CNN_Model, GECNN_Model_1, GECNN_Model_2]

# Check whether a model directory already exists, if not creat one
model_directory = curr_dir + "/Model_MNIST_Parameters"

if not Path(model_directory).exists():
    Path(model_directory).mkdir(parents=True, exist_ok=True)
    print("Model_MNIST-directory is created")
else:
    print("Model_MNIST-directory already exists")

# Check whether a directory for the training losses already exists, if not creat one
model_loss_directory = curr_dir + "/Model_MNIST_Losses"

if not Path(model_loss_directory).exists():
    Path(model_loss_directory).mkdir(parents=True, exist_ok=True)
    print("Model_MNIST_loss-directory is created")
else:
    print("Model_MNIST_loss-directory already exists")

# Check whether a directory for the training losses already exists, if not creat one
model_accuarcy_directory = curr_dir + "/Model_MNIST_accuracy"

if not Path(model_accuarcy_directory).exists():
    Path(model_accuarcy_directory).mkdir(parents=True, exist_ok=True)
    print("Model_MNIST_accuracy-directory is created")
else:
    print("Model_MNIST_accuaracy-directory already exists")

# Start the training loop for each model which is not trained already
for model in Models:
    
    # Check whether a trained model 1 already exists, if not start training
    model_path = Path(model_directory) / str(model).split("(")[0]

    model_loss_path = Path(model_loss_directory) / str(model).split("(")[0]

    model_accuracy_path = Path(model_accuarcy_directory) / str(model).split("(")[0]
    
    if not Path(model_path).exists():
        print(f"No trained model of {str(model).split('(')[0]} exists. The training of this model will start.")
        train(model, epochs=18, lr=0.001, dataloader= train_dataloader, model_path=model_path, model_loss_path=model_loss_path, model_accuracy_path=model_accuracy_path)
    else:
        print(f"An already trained model of {str(model).split('(')[0]} exists.")
    
        