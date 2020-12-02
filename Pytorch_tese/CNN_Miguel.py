#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MyDataset import MyDataset


#Create CNN
class CNN(nn.Module):
    #self.conv1-> Kernel size sets the filter size, out_channels sets the number of filters
    #self.fc1-> out_features sets the size of the output tensor
    def __init__(self, in_channels=1, num_classes=10):         #in_channels=color of image, num_classes=number of features
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))      #fist convulotional layer
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))                                                  #polling
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=num_classes)                                         #fuly connected layer, 7*7 are the higth and width of each of 16 output channels 

    
    def forward(self, x):
        #hidden conv layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        #hidden conv layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #hidden linear layer
        x = x.reshape(-1, 16*7*7)
        x = self.fc1(x)

        return x                                                                                

#model = CNN()
#x = torch.randn(64, 1, 28, 28)
#print(model.forward(x).shape)
#print(model(x).shape)
#exit()

#set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameteres
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1 

#load data
dataset = MyDataset(csv_file="example.csv", dir="example", transform=transforms.ToTensor())
train_set, test_set = torch.utils.ramdom_split(dataset, [1000, 200])     #divide the dataset in 1000  for train_set and 200 to test_set
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#iniatialize network
model = CNN().to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train network