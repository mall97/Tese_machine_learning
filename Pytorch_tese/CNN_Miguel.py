#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


#Create CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):         #in_channels=color of image
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))      #fist convulotional layer
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))                                                  #polling
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)                                                                                      #fuly connected layer

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x                                                                                

model = CNN()
x = torch.randn(64, 1, 28, 28)
print(model.forward(x).shape)
print(model(x).shape)
exit()

#set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameteres

