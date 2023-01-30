import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5)) # haven't mentioned stride and padding size as they are set default to 1 and 0 respectively
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5))
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)
    def forward(self,x):
        x = torch.reshape(x, (x.shape[0],1,32,32))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0],120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # now the output size is 64(batch_size)x10
        return x