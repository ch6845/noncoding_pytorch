import torch.nn as nn
import torch.nn.functional as F


class DeepSEA(nn.Module):
    def __init__(self):
        super(DeepSEA, self).__init__()
        
        self.batch_first=True
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(53*960, 925)
        self.fc2 = nn.Linear(925, 919)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.drop1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = x.view(-1, 53*960)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x