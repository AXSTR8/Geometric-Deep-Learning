import torch
import torch.nn as nn

class CNN_1(nn.Module):
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
    


class CNN_2(nn.Module):
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
    

class CNN_3(nn.Module):
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