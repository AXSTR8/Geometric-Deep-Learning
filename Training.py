import os
import torch
import torch.nn as nn
import numpy as np
from Models import CNN_1, CNN_2, CNN_3
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Create the Convolutional Neural Network Model
CNN_1_Model = CNN_1()
CNN_2_Model = CNN_2()
CNN_3_Model = CNN_3()

# Use the GPU for a faster training process if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CNN_1_Model.to(device)
CNN_2_Model.to(device)
CNN_3_Model.to(device)

# Define the current directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

# Check whether a model directory already exists, if not creat one
model_directory = curr_dir + "/Model_Parameters"
model_dir_path = Path(model_directory)
model_1_path = model_directory + "\CNN_1_Model"
model_2_path = model_directory + "\CNN_2_Model"
model_3_path = model_directory + "\CNN_3_Model"

if not os.path.isdir(model_dir_path):
    model_dir_path.mkdir(parents=True, exist_ok=True)

# Check whether a trained model already exists, if not start training 
if not os.path.isfile(model_1_path):

    # Determine the hyperparameters for the training
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    
    # Specify the path where the training data is stored
    train_data_dir = curr_dir + "\Data\Train"
    train_data_path = Path(train_data_dir)
    print(train_data_path)

    # Define the data-transformer to convert the image data into PyTorch Tensors of a specified dimension
    data_transformer = transforms.Compose([
        transforms.Resize(size=(256,256)), #Resize the image to 256x256
        transforms.ToTensor()
    ])

    # Create the training data
    train_data = datasets.ImageFolder(root=train_data_dir, transform=data_transformer, target_transform=None)

    # Create a train DataLoader:
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)

    # Define the optimizer and the loss function and create a loss tracking list
    optimizer = torch.optim.Adam(CNN_1_Model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_tracker = []

    # Define the training loop
    for epoch in range(epochs):
        for i, (images, targets) in enumerate(train_dataloader):
            
            # Push the images and targets to the device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = CNN_1_Model.forward(images)
            loss = criterion(outputs, targets)
            loss_tracker.append(loss.cpu().detach().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Report over the training progress after certain epoch
        print(f"CNN_1_Model: In epoch {epoch+1} with loss = {loss.item()}.")
    
    torch.save(CNN_1_Model.state_dict(), model_1_path)
    


if not os.path.isfile(model_2_path):

    # Determine the hyperparameters for the training
    batch_size = 16
    epochs = 10
    learning_rate = 0.12
    
    # Specify the path where the training data is stored
    train_data_dir = curr_dir + "\Data\Train"
    train_data_path = Path(train_data_dir)
    print(train_data_path)

    # Define the data-transformer to convert the image data into PyTorch Tensors of a specified dimension
    data_transformer = transforms.Compose([
        transforms.Resize(size=(256,256)), #Resize the image to 256x256
        transforms.ToTensor()
    ])

    # Create the training data
    train_data = datasets.ImageFolder(root=train_data_dir, transform=data_transformer, target_transform=None)

    # Create a train DataLoader:
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)

    # Define the optimizer and the loss function and create a loss tracking list
    optimizer = torch.optim.Adam(CNN_2_Model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_tracker = []

    # Define the training loop
    for epoch in range(epochs):
        for i, (images, targets) in enumerate(train_dataloader):
            
            # Push the images and targets to the device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = CNN_2_Model.forward(images)
            loss = criterion(outputs, targets)
            loss_tracker.append(loss.cpu().detach().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Report over the training progress after certain epoch
        print(f"CNN_2_Model: In epoch {epoch+1} with loss = {loss.item()}.")
    
    torch.save(CNN_2_Model.state_dict(), model_2_path)

#Model_Layers = [module for module in CNN_2_Model.modules() if not isinstance(module, nn.Sequential)]

#print(Model_Layers)




# Check whether a trained model already exists, if not start training 
if not os.path.isfile(model_3_path):

    # Determine the hyperparameters for the training
    batch_size = 32
    epochs = 10
    learning_rate = 0.01
    
    # Specify the path where the training data is stored
    train_data_dir = curr_dir + "\Data\Train"
    train_data_path = Path(train_data_dir)
    print(train_data_path)

    # Define the data-transformer to convert the image data into PyTorch Tensors of a specified dimension
    data_transformer = transforms.Compose([
        transforms.Resize(size=(256,256)), #Resize the image to 256x256
        transforms.ToTensor()
    ])

    # Create the training data
    train_data = datasets.ImageFolder(root=train_data_dir, transform=data_transformer, target_transform=None)

    # Create a train DataLoader:
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)

    # Define the optimizer and the loss function and create a loss tracking list
    optimizer = torch.optim.Adam(CNN_3_Model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_tracker = []

    # Define the training loop
    for epoch in range(epochs):
        for i, (images, targets) in enumerate(train_dataloader):
            
            # Push the images and targets to the device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = CNN_3_Model.forward(images)
            loss = criterion(outputs, targets)
            loss_tracker.append(loss.cpu().detach().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Report over the training progress after certain epoch
        print(f"CNN_3_Model: In epoch {epoch+1} with loss = {loss.item()}.")
    
    torch.save(CNN_3_Model.state_dict(), model_3_path)
    