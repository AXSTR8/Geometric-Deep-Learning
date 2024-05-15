import torch
import torch.nn as nn
import numpy as np
try:
    import escnn
    from escnn import gspaces
    from escnn import nn as nn_str
except:
    pass


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
    def __init__(self, num_out: int):
        super().__init__()
        
        # The first Convolutional block 
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3,10,4),
                                          nn.BatchNorm2d(10),
                                          nn.ReLU())
        
        # A pooling layer
        self.pl1 = nn.AvgPool2d(4)
        
        # The second Convolutional block 
        self.conv_block_2 = nn.Sequential(nn.Conv2d(10,22,5),
                                          nn.BatchNorm2d(22),
                                          nn.ReLU())
        
        # A pooling layer
        self.pl2 = nn.AvgPool2d(4)

        # The third Convolutional block 
        self.conv_block_3 = nn.Sequential(nn.Conv2d(22,20,4),
                                          nn.BatchNorm2d(20),
                                          nn.ReLU())
        
        # A pooling layer
        self.pl3 = nn.AvgPool2d(3)

        # The fourth Convolutional block 
        self.conv_block_4 = nn.Sequential(nn.Conv2d(20,20,3),
                                          nn.BatchNorm2d(20),
                                          nn.ReLU())
        
        # Fully Connected Neural Network
        self.fc_block = nn.Sequential(nn.Linear(20, 55),
                                    nn.ReLU(),
                                    nn.Linear(55, num_out))
    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.pl1(out)
        out = self.conv_block_2(out)
        out = self.pl2(out)
        out = self.conv_block_3(out)
        out = self.pl3(out)
        out = self.conv_block_4(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc_block(out)
        return out
    
    def prediction(self, x):
            pred = self.forward(x)
            return torch.argmax(pred)

try:
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
            self.conv_block_1 = nn_str.SequentialModule(nn_str.R2Conv(self.input_field, output_field_block_1, kernel_size=4),
                                                        nn_str.InnerBatchNorm(output_field_block_1),
                                                        nn_str.ReLU(output_field_block_1))
            
            # Add a pooling layer 
            self.pl_1 = nn_str.PointwiseAvgPool(output_field_block_1, 4)

            # The second Convolutional block 
            # The input field type is the output field type of the previous layer
            input_field_block_2 = self.conv_block_1.out_type
           
            # Determine the output field type
            output_field_block_2 = nn_str.FieldType(self.r2_space, 22*[self.r2_space.regular_repr])
            self.conv_block_2 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_2, output_field_block_2, kernel_size=5),
                                                        nn_str.InnerBatchNorm(output_field_block_2),
                                                        nn_str.ReLU(output_field_block_2))
            
            # Add a pooling layer
            self.pl_2 = nn_str.PointwiseAvgPool(output_field_block_2, 4)

            # The third Convolutional block 
            # The input field type is the output field type of the previous layer
            input_field_block_3 = self.conv_block_2.out_type
           
            # Determine the output field type
            output_field_block_3 = nn_str.FieldType(self.r2_space, 20*[self.r2_space.regular_repr])
            self.conv_block_3 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_3, output_field_block_3, kernel_size=4),
                                                        nn_str.InnerBatchNorm(output_field_block_3),
                                                        nn_str.ReLU(output_field_block_3))
            
            # Add a pooling layer
            self.pl_3 = nn_str.PointwiseAvgPool(output_field_block_3, 3)

            # The fourth Convolutional block 
            # The input field type is the output field type of the previous layer
            input_field_block_4 = self.conv_block_3.out_type
            
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
except:
     pass