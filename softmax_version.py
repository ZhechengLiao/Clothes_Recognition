"""
Use a simple softmax neural network to do the clothes recognition
"""
# import dependencies
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Load datasets
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data,batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=32, shuffle=False)

# Define Basic Model
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.flatten = nn.Flatten() # flatten input first
        self.linear1 = nn.Linear(784, 64) # flatten to (Batchsize, flatten size), flatten size, output size
        self.linear2 = nn.Linear(64, 10)  # output layer 
        
    def __call__(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# train
def train(num_epochs):
    """Train the model in number of epochs"""
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad() # ****important****
            outputs = model(inputs.squeeze()) # squeeze here is important, to eleminate 1 dimention
            loss = loss_fn(outputs, labels)   # here, we have to make sure the shape of outputs and shape of labels is same
            loss.backward() # backward propagation
            optimizer.step()
        print(loss)

def accuracy(loader, model):
    """get the accuracy of model"""
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x
            y = y
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()

model = ClassificationModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(num_epochs=10)
accuracy(test_dataloader, model)

"""
Result: 
tensor(0.4362, grad_fn=<NllLossBackward0>)
tensor(0.2650, grad_fn=<NllLossBackward0>)
tensor(0.7667, grad_fn=<NllLossBackward0>)
tensor(0.2823, grad_fn=<NllLossBackward0>)
tensor(0.3702, grad_fn=<NllLossBackward0>)
tensor(0.2060, grad_fn=<NllLossBackward0>)
tensor(0.1329, grad_fn=<NllLossBackward0>)
tensor(0.1998, grad_fn=<NllLossBackward0>)
tensor(0.1506, grad_fn=<NllLossBackward0>)
tensor(0.4562, grad_fn=<NllLossBackward0>)
Got 8813 / 10000 with accuracy 88.13
"""