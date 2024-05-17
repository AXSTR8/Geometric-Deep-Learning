import os
import torch
import torch.nn as nn
import numpy as np
from Models import CNN_1, CNN_2, CNN_3, C16_Equivariant_CNN, SO2SteerableCNN_B
from pathlib import Path
from wslpath import wslpath
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# Determine the hyperparameters for the training
batch_size = 32
epochs = 10
learning_rate = 0.001

# Specify the current directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

# Specify the path where the training data is stored
train_data_dir = curr_dir + "/Data/Train"
train_data_path = Path(train_data_dir)

# Define the data-transformer to convert the image data into PyTorch Tensors of a specified dimension
data_transformer = transforms.Compose([
    transforms.Resize(size=(128,128)), #Resize the image to 128x128
    transforms.ToTensor()
])

# Import the training data either as path on a Windows or a Linux Subsystem Path
try:
    train_data = datasets.ImageFolder(root=wslpath(train_data_path), transform=data_transformer, target_transform=None)
except:
    train_data = datasets.ImageFolder(root=train_data_path, transform=data_transformer, target_transform=None)

# Create a train DataLoader:
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
        
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

        test_accuracy = test(model, train_dataloader)
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
def test(model: torch.nn.Module, dataloader= train_dataloader):
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

# Create the Convolutional Neural Network Model
CNN_1_Model = CNN_1()
CNN_2_Model = CNN_2()
CNN_3_Model = CNN_3(6)

# Create the C16 Equivaraint CNN
ECNN_1_Model = C16_Equivariant_CNN(6)
ECNN_2_Model = SO2SteerableCNN_B()

# Use the GPU for a faster training process if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CNN_1_Model.to(device)
CNN_2_Model.to(device)
CNN_3_Model.to(device)
ECNN_1_Model.to(device)
ECNN_2_Model.to(device)

# Create a list of all CNN Models
Models_list = [CNN_1_Model, CNN_2_Model, CNN_3_Model, ECNN_1_Model, ECNN_2_Model]

# Check whether a model directory already exists, if not creat one
model_directory = curr_dir + "/Model_Parameters"

# Create the directory if it not already exists
if not Path(model_directory).exists():
    Path(model_directory).mkdir(parents=True, exist_ok=True)
    print("Model-directory is created")
else:
    print("Model-directory already exists")

# Check whether a directory for the training losses already exists, if not creat one
model_loss_directory = curr_dir + "/Model_Losses"

# Create the directory if it not already exists
if not Path(model_loss_directory).exists():
    Path(model_loss_directory).mkdir(parents=True, exist_ok=True)
    print("Model-loss-directory is created")
else:
    print("Model-loss-directory already exists")


# Check whether a model directory already exists, if not creat one
model_directory = curr_dir + "/Model_Parameters"

if not Path(model_directory).exists():
    Path(model_directory).mkdir(parents=True, exist_ok=True)
    print("Model-directory is created")
else:
    print("Model-directory already exists")

# Check whether a directory for the training losses already exists, if not creat one
model_loss_directory = curr_dir + "/Model_Losses"

if not Path(model_loss_directory).exists():
    Path(model_loss_directory).mkdir(parents=True, exist_ok=True)
    print("Model_loss-directory is created")
else:
    print("Model_loss-directory already exists")

# Check whether a directory for the training losses already exists, if not creat one
model_accuarcy_directory = curr_dir + "/Model_accuracy"

if not Path(model_accuarcy_directory).exists():
    Path(model_accuarcy_directory).mkdir(parents=True, exist_ok=True)
    print("Model_accuracy-directory is created")
else:
    print("Model_accuaracy-directory already exists")

# Start the training loop for each model which is not trained already
for model in Models_list:
    
    # Check whether a trained model 1 already exists, if not start training
    model_path = Path(model_directory) / str(model).split("(")[0]

    model_loss_path = Path(model_loss_directory) / str(model).split("(")[0]

    model_accuracy_path = Path(model_accuarcy_directory) / str(model).split("(")[0]
    
    if not Path(model_path).exists():
        print(f"No trained model of {str(model).split('(')[0]} exists. The training of this model will start.")
        train(model, epochs=10, lr=0.001, dataloader= train_dataloader, model_path=model_path, model_loss_path=model_loss_path, model_accuracy_path=model_accuracy_path)
    else:
        print(f"An already trained model of {str(model).split('(')[0]} exists.")
    
        