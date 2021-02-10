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
    def __init__(self, in_channels=3, num_classes=10):         #in_channels=color of image (node number), num_classes=number of features
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))      #fist convulotional layer
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))                                                  #polling
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(6,6))
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=144, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(num_features=144)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(6,6))

        self.fc1 = nn.Linear(in_features=144*1*1, out_features=100)                                         #fuly connected layer, 7*7 are the higth and width of each of 16 output channels 
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=16*3*3, out_features=100)
        self.out = nn.Linear(in_features=50, out_features=num_classes)
        self.out2 = nn.Linear(in_features=100, out_features=num_classes)
        self.drop = torch.nn.Dropout(0.2)

    
    def forward(self, x):
        #hidden conv layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        #hidden conv layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        #hidden conv layer
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        #hidden linear layers
        x = x.reshape(-1, 144*1*1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.out(x)

        return x                                                                                

    def forward2(self, x):
        #hidden conv layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        #hidden conv layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        #hidden linear layers
        x = x.reshape(-1, 16*3*3)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.out(x)

        return x

    def forward3(self, x):
        #hidden conv layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        #hidden conv layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        #hidden linear layers
        x = x.reshape(-1, 16*3*3)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out2(x)

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
batch_size = 32
num_epochs = 80 

#load data
#dataset = MyDataset(csv_file='C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\data.csv', dir='C:\\Users\\Miguel\\Desktop\\Tese_machine_learning\\new_size', transform=transforms.ToTensor())
#train_set, test_set = torch.utils.data.random_split(dataset, [600,79])     #divide the dataset in 1000  for train_set and 200 to test_set
#train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#cifar
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download='True', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=cifar_trainset, batch_size=batch_size, shuffle=True)

cifar_testset = datasets.CIFAR10(root='./data', train=False, download='True', transform=transforms.ToTensor())
test_loader = DataLoader(dataset=cifar_testset, batch_size=batch_size, shuffle=True)

#iniatialize network
model = CNN().to(device)
#loss and optimizer
criterion = nn.CrossEntropyLoss()                                # is used for multiclass classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)    #implements various optimization algorithms (Stochastic Optimization)

#Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):      #train loader divide by image and classification (data=image, targets=classification), batch_idx=number of the cycle
        #get data to cuda
        data = data.to(device=device)
        targets =targets.to(device=device)
        
        #forward ->compute the output during forward pass
        scores = model(data)
        loss = criterion(scores, targets)     

        #backward ->compute the gradient to be propagated
        optimizer.zero_grad()
        loss.backward()

        #gradient descent 
        optimizer.step()

#file= 'model.pth'
#torch.save(model, file)

#model=torch.load(file)     #load the model saved
#model.eval()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device) 
            y = y.to(device=device)

            scores = model(x) 
            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
            print(num_correct, num_samples)
            print(f'{float(num_correct)/float(num_samples)*100:.2f}')
    model.train()

#check_accuracy(train_loader, model)
check_accuracy(test_loader, model)