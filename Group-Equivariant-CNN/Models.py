import torch
import escnn
import torch.nn as nn
import numpy as np
from escnn import gspaces
from escnn import nn as nn_str


class CNN_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Conv2d(3,32,3)
        self.rl = nn.ReLU()
        self.pl1 = nn.AvgPool2d(2)
        self.cn2 = nn.Conv2d(32,16,3)
        self.pl2 = nn.AvgPool2d(3)
        self.cn3 = nn.Conv2d(16,1,3)
        self.pl3 = nn.AvgPool2d(4)
        self.ll1 = nn.Linear(81,6)
    
    def forward(self, x):
        out = self.cn1(x)
        out = self.rl(out)
        out = self.pl1(out)
        out = self.cn2(out)
        out = self.rl(out)
        out = self.pl2(out)
        out = self.cn3(out)
        out = self.rl(out)
        out = self.pl3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.ll1(out)
        return out
    
    def prediction(self, x):
            pred = self.forward(x)
            return torch.argmax(pred)
    


class CNN_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Conv2d(3,32,40)
        self.rl = nn.ReLU()
        self.pl1 = nn.AvgPool2d(2)
        self.cn2 = nn.Conv2d(32,16,4)
        self.pl2 = nn.AvgPool2d(3)
        self.cn3 = nn.Conv2d(16,1,3)
        self.pl3 = nn.AvgPool2d(4)
        self.ll1 = nn.Linear(64,6)
    
    def forward(self, x):
        out = self.cn1(x)
        out = self.rl(out)
        out = self.pl1(out)
        out = self.cn2(out)
        out = self.rl(out)
        out = self.pl2(out)
        out = self.cn3(out)
        out = self.rl(out)
        out = self.pl3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.ll1(out)
        return out
    
    def prediction(self, x):
            pred = self.forward(x)
            return torch.argmax(pred)
    

class CNN_3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Conv2d(3,16,12)
        self.rl = nn.ReLU()
        self.cn2 = nn.Conv2d(16,16,3)
        self.pl1 = nn.MaxPool2d(4)
        self.cn3 = nn.Conv2d(16,16,3)
        self.cn4 = nn.Conv2d(16,16,3)
        self.pl2 = nn.MaxPool2d(2)
        self.cn5 = nn.Conv2d(16,16,3)
        self.cn6 = nn.Conv2d(16,1,3)
        self.pl2 = nn.AvgPool2d(3)
        self.ll1 = nn.Linear(256,6)
    
    def forward(self, x):
        out = self.cn1(x)
        out = self.rl(out)
        out = self.cn2(out)
        out = self.rl(out)
        out = self.pl1(out)
        out = self.cn3(out)
        out = self.rl(out)
        out = self.cn4(out)
        out = self.rl(out)
        out = self.pl2(out)
        out = self.cn6(out)
        out = torch.flatten(out, start_dim=1)
        out = self.ll1(out)
        return out
    
    def prediction(self, x):
            pred = self.forward(x)
            return torch.argmax(pred)


class C16_Equivariant_CNN(torch.nn.Module):
    def __init__(self, num_out : int):
        super().__init__()
        # Define the base_group to be C16
        self.r2_space = gspaces.rot2dOnR2(N=16)
        
        # The input field type are 3 trivial representations, as the input image has three color channels
        self.input_field = nn_str.FieldType(self.r2_space, 3*[self.r2_space.trivial_repr])
         
        # The first Convolutional block 
        # Determine the output field type
        output_field_block_1 = nn_str.FieldType(self.r2_space, 10*[self.r2_space.regular_repr])
        self.conv_block_1 = nn_str.SequentialModule(nn_str.R2Conv(self.input_field, output_field_block_1, kernel_size=7),
                                                    nn_str.InnerBatchNorm(output_field_block_1),
                                                    nn_str.ReLU(output_field_block_1))
        # Add a pooling layer 
        self.pl_1 = nn_str.PointwiseAvgPool(output_field_block_1, 4)


        # The second Convolutional block 
        # The input field type is the output field type of the previous layer
        input_field_block_2 = self.conv_block_1.out_type
        # Determine the output field type
        output_field_block_2 = nn_str.FieldType(self.r2_space, 22*[self.r2_space.regular_repr])
        self.conv_block_2 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_2, output_field_block_2, kernel_size=7),
                                                    nn_str.InnerBatchNorm(output_field_block_2),
                                                    nn_str.ReLU(output_field_block_2))
        # Add a pooling layer
        self.pl_2 = nn_str.PointwiseAvgPool(output_field_block_2, 5)

        # The third Convolutional block 
        # The input field type is the output field type of the previous layer
        input_field_block_3 = self.conv_block_2.out_type
        # Determine the output field type
        output_field_block_3 = nn_str.FieldType(self.r2_space, 20*[self.r2_space.regular_repr])
        self.conv_block_3 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_3, output_field_block_3, kernel_size=7),
                                                    nn_str.InnerBatchNorm(output_field_block_3),
                                                    nn_str.ReLU(output_field_block_3))
        # Add a pooling layer
        self.pl_3 = nn_str.PointwiseAvgPool(output_field_block_3, 4)

        # The fourth Convolutional block 
        # The input field type is the output field type of the previous layer
        input_field_block_4 = self.conv_block_2.out_type
        # Determine the output field type
        output_field_block_4 = nn_str.FieldType(self.r2_space, 20*[self.r2_space.regular_repr])
        self.conv_block_4 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_4, output_field_block_4, kernel_size=3),
                                                    nn_str.InnerBatchNorm(output_field_block_4),
                                                    nn_str.ReLU(output_field_block_4))


        # Perform the Pooling over the complete group
        self.gpool = nn_str.GroupPooling(output_field_block_4)

        # Fully Connected Neural Network
        self.fc_block = nn.Sequential(nn.Linear(320, 55),
                                      nn.ReLU(),
                                      nn.Linear(55, num_out))
        
    
    def forward(self, x):
        out = nn_str.GeometricTensor(x, self.input_field)
        out = self.conv_block_1(out)
        out = self.pl_1(out)
        out = self.conv_block_2(out)
        out = self.pl_2(out)
        out = self.conv_block_3(out)
        out = self.pl_3(out)
        out = self.conv_block_4(out)
        out = out.tensor
        out = self.fc_block(out.reshape(out.shape[0], -1))
        return out
    
    def prediction(self, x):
            pred = self.forward(x)
            return torch.argmax(pred)

#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))
