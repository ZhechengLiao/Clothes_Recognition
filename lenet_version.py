"""Build a LeNet5 to do image recognition in FashionMNIST datasets"""

# Add dependencies
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

# Get Datasets
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.Resize((32, 32)), # because LeNet is for 32*32 image, so we change these datas to 32*32
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std = (0.3081,))
    ])
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=False,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1325,), std = (0.3105,))
    ])
)

# Data Loader
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Build Model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # conv layer
        self.convLayer1 = nn.Sequential(
            # number of channel input, number of channel output, kernelsize
            nn.Conv2d(1, 6, 5), 
            nn.BatchNorm2d(6),
            nn.Tanh(),
            # kernel size 2*2, stride: 2
            nn.MaxPool2d(2, stride=2),
        )
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5), 
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2), 
        )
        
        # fully connect layer
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )
        
        # flatten layer
        self.flatten = nn.Flatten()
        
    def __call__(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# train
def train(num_epochs):
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch:{epoch+1}, Loss:{loss}')

# test
def test():
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            
        print(f'Accuracy: {100*correct/total}')

# main
model = LeNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train(10)
test()

"""
Result:
Epoch:1, Loss:0.5358776450157166
Epoch:2, Loss:0.7002905607223511
Epoch:3, Loss:0.41531533002853394
Epoch:4, Loss:0.32789814472198486
Epoch:5, Loss:0.36516523361206055
Epoch:6, Loss:0.3309135437011719
Epoch:7, Loss:0.4501449465751648
Epoch:8, Loss:0.5225855708122253
Epoch:9, Loss:0.26034900546073914
Epoch:10, Loss:0.3931770622730255
Accuracy: 87.96
"""