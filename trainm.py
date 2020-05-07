import cv2                
import matplotlib.pyplot as plt    	
import torchvision.models as models
import torch.nn as nn
import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from glob import glob
from PIL import Image
train_data_path = 'Data/'
val_data_path = 'Data/'

train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
            ])

val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
            ])

train_data = datasets.ImageFolder(train_data_path, transform=train_transform)
val_data = datasets.ImageFolder(val_data_path, transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)


loaders_transfer = {'train': train_loader, 'valid': val_loader}
softmax = nn.Softmax(dim=1)
#model_transfer = models.resnet50(pretrained=True)
#model_transfer.fc = nn.Linear(2048, 2)
use_cuda = torch.cuda.is_available()
#if use_cuda:
#    model_transfer = model_transfer.cuda()
import torch.nn as nn
import torch.nn.functional as F

softmax = nn.Softmax(dim=1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding = 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding = 2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool =  nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(28*28*128, 256)
        self.fc2 = nn.Linear(256 ,128)
        self.output = nn.Linear(128,2)
        
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        ## Define forward behavior
        x  = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x  = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x  = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(-1, 14*14*256)
        x = F.relu(self.fc1(self.drop(x)))
        x = F.relu(self.fc2(self.drop(x)))
        x =(self.output(self.drop(x)))
        
        return x

model_transfer = Net()

if use_cuda:
    model_transfer.cuda()

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(model_transfer.parameters(),lr=0.003)

losses = {'train':[], 'validation':[]}
losses = {'train':[], 'validation':[]}

print('training to be strted' , use_cuda)
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        
        
        print('training started..............')
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            
            pred = model(data)
            loss = criterion(pred, target)
            
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            loss.backward()
            optimizer.step()
        model.eval()
        
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            val_pred = model(data)
            val_loss = criterion(val_pred, target)
            
            valid_loss += ((1 / (batch_idx + 1)) * (val_loss.data - valid_loss))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
            
        if (valid_loss < valid_loss_min):
            print("Saving model.  Validation loss:... {} --> {}".format(valid_loss_min, valid_loss.item()))
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
            print()
            
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)
        
        
    return model

def test(loaders, model, criterion, use_cuda):

    test_loss = 0.
    correct = 0.
    total = 0.
    
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))





#model_transfer = train(20, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model.pt')
#model_transfer.load_state_dict(torch.load('model.pt'))
#test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
