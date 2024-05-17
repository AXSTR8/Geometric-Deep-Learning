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
        self.ll1 = nn.Linear(16,6)
    
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
        self.ll1 = nn.Linear(4,6)
    
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
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3,32,3),
                                          nn.ReLU())
        
        # A pooling layer
        self.pl1 = nn.AvgPool2d(2)
        
        # The second Convolutional block 
        self.conv_block_2 = nn.Sequential(nn.Conv2d(32,64,3),
                                          nn.ReLU())
        
        # A pooling layer
        self.pl2 = nn.AvgPool2d(3)

        # The third Convolutional block 
        self.conv_block_3 = nn.Sequential(nn.Conv2d(64,32,3),
                                          nn.ReLU())
        
        # A pooling layer
        self.pl3 = nn.AvgPool2d(3)

        # The fourth Convolutional block 
        self.conv_block_4 = nn.Sequential(nn.Conv2d(32,16,3),
                                          nn.ReLU())
        
        
        # Fully Connected Neural Network
        self.fc_block = nn.Sequential(nn.Linear(256, 55),
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
            output_field_block_1 = nn_str.FieldType(self.r2_space, 32*[self.r2_space.trivial_repr])
            self.conv_block_1 = nn_str.SequentialModule(nn_str.R2Conv(self.input_field, output_field_block_1, kernel_size=4),
                                                        #nn_str.InnerBatchNorm(output_field_block_1),
                                                        nn_str.ReLU(output_field_block_1))
            
            # Add a pooling layer 
            self.pl_1 = nn_str.PointwiseAvgPool(output_field_block_1, 3)

            # The second Convolutional block 
            # The input field type is the output field type of the previous layer
            input_field_block_2 = output_field_block_1
           
            # Determine the output field type
            output_field_block_2 = nn_str.FieldType(self.r2_space, 64*[self.r2_space.trivial_repr])
            self.conv_block_2 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_2, output_field_block_2, kernel_size=5),
                                                        #nn_str.InnerBatchNorm(output_field_block_2),
                                                        nn_str.ReLU(output_field_block_2))
            
            # Add a pooling layer
            self.pl_2 = nn_str.PointwiseAvgPool(output_field_block_2, 3)

            # The third Convolutional block 
            # The input field type is the output field type of the previous layer
            input_field_block_3 = output_field_block_2
           
            # Determine the output field type
            output_field_block_3 = nn_str.FieldType(self.r2_space, 32*[self.r2_space.trivial_repr])
            self.conv_block_3 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_3, output_field_block_3, kernel_size=4),
                                                        #nn_str.InnerBatchNorm(output_field_block_3),
                                                        nn_str.ReLU(output_field_block_3))
            
            # Add a pooling layer
            self.pl_3 = nn_str.PointwiseAvgPool(output_field_block_3, 3)

            # The fourth Convolutional block 
            # The input field type is the output field type of the previous layer
            input_field_block_4 = output_field_block_3
            
            # Determine the output field type
            output_field_block_4 = nn_str.FieldType(self.r2_space, 16*[self.r2_space.trivial_repr])
            self.conv_block_4 = nn_str.SequentialModule(nn_str.R2Conv(input_field_block_4, output_field_block_4, kernel_size=3),
                                                        #nn_str.InnerBatchNorm(output_field_block_4),
                                                        nn_str.ReLU(output_field_block_4))

            # Perform the Pooling over the complete group
            self.gpool = nn_str.GroupPooling(output_field_block_4)

            # Fully Connected Neural Network
            self.fc_block = nn.Sequential(nn.Linear(16, 20),
                                        nn.ReLU(),
                                        nn.Linear(20, num_out))
            
        
        def forward(self, x):
            out = nn_str.GeometricTensor(x, self.input_field)
            out = self.conv_block_1(out)
            out = self.pl_1(out)
            out = self.conv_block_2(out)
            out = self.pl_2(out)
            out = self.conv_block_3(out)
            out = self.pl_3(out)
            out = self.conv_block_4(out)
            out = self.gpool(out)
            out = out.tensor
            out = self.fc_block(out.reshape(out.shape[0], -1))
            return out
        
        def prediction(self, x):
                pred = self.forward(x)
                return torch.argmax(pred)

except:
     pass


try:
    class SO2_Equivariant_CNN(torch.nn.Module):
        def __init__(self, num_out : int):
            super().__init__()
            # Define the base_group to be C16
            self.r2_space = gspaces.rot2dOnR2(N=-1)
            
            # The input field type are 3 trivial representations, as the input image has three color channels
            self.input_field = nn_str.FieldType(self.r2_space, 3*[self.r2_space.trivial_repr])

            self.non_linear_type = nn_str.FourierELU(self.r2_space, 16, irreps=self.r2_space.fibergroup.bl_irreps(2), N=12)
            
            # The first Convolutional block 
            # Determine the output field type
            self.conv_block_1 = nn_str.SequentialModule(nn_str.R2Conv(self.input_field, self.non_linear_type.in_type, kernel_size=4),
                                                        #nn_str.InnerBatchNorm(output_field_block_1),
                                                        self.non_linear_type)
            
            # Add a pooling layer 
            self.pl_1 = nn_str.PointwiseAvgPool(self.non_linear_type.out_type, 3)

            # The second Convolutional block 
            # The input field type is the output field type of the previous layer
            
            # Determine the output field type
            self.conv_block_2 = nn_str.SequentialModule(nn_str.R2Conv(self.non_linear_type.out_type, self.non_linear_type.out_type, kernel_size=5),
                                                        #nn_str.InnerBatchNorm(output_field_block_2),
                                                        self.non_linear_type)
            
            # Add a pooling layer
            self.pl_2 = nn_str.PointwiseAvgPool(self.non_linear_type.out_type, 3)

            # The third Convolutional block 
            self.conv_block_3 = nn_str.SequentialModule(nn_str.R2Conv(self.non_linear_type.out_type, self.non_linear_type.in_type, kernel_size=4),
                                                        #nn_str.InnerBatchNorm(output_field_block_3),
                                                        self.non_linear_type)
            
            # Add a pooling layer
            self.pl_3 = nn_str.PointwiseAvgPool(self.non_linear_type.out_type, 3)

            # The fourth Convolutional block
            self.conv_block_4 = nn_str.SequentialModule(nn_str.R2Conv(self.non_linear_type.out_type, self.non_linear_type.out_type, kernel_size=3),
                                                        #nn_str.InnerBatchNorm(output_field_block_4),
                                                        nn_str.ReLU(self.non_linear_type.out_type))

            # Transforms to the output fields
            #output_invariant_type = nn.FieldType(self.r2_act, 4*[self.r2_act.trivial_repr])
            #self.invariant_map = nn.R2Conv(self.non_linear_type.out_type, output_invariant_type, kernel_size=1, bias=False)

            # Fully Connected Neural Network
            self.fc_block = nn.Sequential(nn.Linear(400, 20),
                                        nn.ReLU(),
                                        nn.Linear(20, num_out))
            
        
        def forward(self, x):
            out = nn_str.GeometricTensor(x, self.input_field)
            out = self.conv_block_1(out)
            out = self.pl_1(out)
            out = self.conv_block_2(out)
            out = self.pl_2(out)
            out = self.conv_block_3(out)
            out = self.pl_3(out)
            out = self.conv_block_4(out)
            out = self.gpool(out)
            out = out.tensor
            out = self.fc_block(out.reshape(out.shape[0], -1))
            return out
        
        def prediction(self, x):
                pred = self.forward(x)
                return torch.argmax(pred)

except:
     pass

class SO2SteerableCNN_B(torch.nn.Module):

    def __init__(self, n_classes=6):

        super().__init__()

        # The model is equivariant under all planar rotations
        self.r2_act = gspaces.rot2dOnR2(N=-1)

        # The group G = SO(2)
        self.G = self.r2_act.fibergroup

        # The input image are three scalar fields, corresponding to three trivial representation
        in_type = nn_str.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])

        # The input type for wrapping the images into a geometric tensors is stored during the forward pass
        self.input_type = in_type

        # The first convolution
        # 8 feature fields, each transforming under the regular representation of SO(2) up to frequency 3
        # When taking the ELU non-linearity, we sample the feature fields on N=16 points
        activation1 = nn_str.FourierELU(self.r2_act, 8, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation1.in_type
        self.block1 = nn_str.SequentialModule(
            nn_str.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn_str.IIDBatchNorm2d(out_type),
            activation1,
        )

        # The second convolution 
        # The output type of the previous layer is the input type to the next layer
        in_type = self.block1.out_type
        # The output type of the second convolution layer are 16 regular feature fields
        activation2 = nn_str.FourierELU(self.r2_act, 16, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation2.in_type
        self.block2 = nn_str.SequentialModule(
            nn_str.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            nn_str.IIDBatchNorm2d(out_type),
            activation2
        )
        # A pooling layer with a kernel of size 4
        self.pool1 = nn_str.SequentialModule(
            nn_str.PointwiseAvgPool(out_type, 4)
        )

        # The third convolution
        # The output type of the previous layer is the input type to the next layer
        in_type = self.block2.out_type
        # The output type of the third convolution layer are 16 regular feature fields
        activation3 = nn_str.FourierELU(self.r2_act, 16, irreps=self.G.bl_irreps(3), N=8, inplace=True)
        out_type = activation3.in_type
        self.block3 = nn_str.SequentialModule(
            nn_str.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn_str.IIDBatchNorm2d(out_type),
            activation3
        )

        # The fourth convolution
        # The output type of the previous layer is the input type to the next layer
        in_type = self.block3.out_type
        # The output type of the fourth convolution layer are 16 regular feature fields
        activation4 = nn_str.FourierELU(self.r2_act, 16, irreps=self.G.bl_irreps(3), N=8, inplace=True)
        out_type = activation4.in_type
        self.block4 = nn_str.SequentialModule(
            nn_str.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            nn_str.IIDBatchNorm2d(out_type),
            activation4
        )
        # Average pooling with kernelsize = 4
        self.pool2 = nn_str.PointwiseAvgPool(out_type, 4)
        

        # The fifth convolution
        # The output type of the previous layer is the input type to the next layer
        in_type = self.block4.out_type
        # The output type of the fifth convolution layer are 16 regular feature fields
        activation5 = nn_str.FourierELU(self.r2_act, 16, irreps=self.G.bl_irreps(3), N=8, inplace=True)
        out_type = activation5.in_type
        self.block5 = nn_str.SequentialModule(
            nn_str.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            nn_str.IIDBatchNorm2d(out_type),
            activation5
        )

        # The sixth convolution
        # The output type of the previous layer is the input type to the next layer
        in_type = self.block5.out_type
        # The output type of the sixth convolution layer are 16 regular feature fields
        activation6 = nn_str.FourierELU(self.r2_act, 16, irreps=self.G.bl_irreps(3), N=8, inplace=True)
        out_type = activation6.in_type
        self.block6 = nn_str.SequentialModule(
            nn_str.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn_str.IIDBatchNorm2d(out_type),
            activation6
        )
        self.pool3 = nn_str.PointwiseAvgPool(out_type, 4)

        # The number of output invariant channels
        c = 256

        # Last 1x1 convolution layer, which maps the regular fields to c=256 invariant scalar fields
        # This is essential to provide *invariant* features in the final classification layer
        output_invariant_type = nn_str.FieldType(self.r2_act, c*[self.r2_act.trivial_repr])
        self.invariant_map = nn_str.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(4*c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(4*c, n_classes),
        )

    def forward(self, input: torch.Tensor):
        # Transform the input Tensor in a GeometricTensor
        x = self.input_type(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.invariant_map(x)

        # Transform the GeometricTensor in a Pytorch Tensor
        x = x.tensor
        x = self.fully_net(x.reshape(x.shape[0], -1))
        return x
    
    def prediction(self, x):
                pred = self.forward(x)
                return torch.argmax(pred)

# Define a function to determine the accuracy on the (train) dataset
def model_accuracy (model, data):
    # Define the device to the GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the number of right and total predictions to zero
    predictions_right = 0
    predictions_total = 0
    model.to(device)
    model.eval()
    for i in range(len(data)):
        # Unsqueeze the 3 dimensional tensor to a 4 dimensional 
        input = torch.unsqueeze(data[i][0], 0)
        input = input.to(device)
        model_prediction = model.prediction(input)
        if model_prediction == data[i][1]:
            predictions_right += 1
        predictions_total += 1
    accuracy = predictions_right / predictions_total
    return accuracy