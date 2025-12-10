# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 00:00:03 2025

@author: Даниш
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
device = torch.device('cpu')
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
train_data = torchvision.datasets.ImageFolder('./data/train', transform=transform)
test_data = torchvision.datasets.ImageFolder('./data/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
num_classes = len(train_data.classes)
print(f"Классы: {train_data.classes}")
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 14 * 14, num_classes) 
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
model1 = MyCNN().to(device)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(10):
    model1.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer1.zero_grad()
        loss = criterion(model1(images), labels)
        loss.backward()
        optimizer1.step()
    print(f'CNN Эпоха {epoch+1}')
model1.eval()
correct1 = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model1(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct1 += (predicted == labels).sum().item()
print(f'CNN Точность: {100*correct1/total:.1f}%')
torch.save(model1.state_dict(), 'my_cnn_model.pth')
'''Предобученная resnet18'''
weights_path = r"C:\Users\Даниш\.cache\torch\hub\checkpoints\resnet18.pth"
model2 = torchvision.models.resnet18(weights=None)
state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
model2.load_state_dict(state_dict)
for param in model2.parameters():
    param.requires_grad = False
model2.fc = nn.Linear(model2.fc.in_features, num_classes)
model2 = model2.to(device)
optimizer2 = torch.optim.Adam(model2.fc.parameters(), lr=0.001)

for epoch in range(15):
    model2.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer2.zero_grad()
        loss = criterion(model2(images), labels)
        loss.backward()
        optimizer2.step()
    print(f'ResNet Эпоха {epoch+1}')
model2.eval()
correct2 = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model2(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct2 += (predicted == labels).sum().item()

print(f'ResNet Точность: {100*correct2/total:.1f}%')
torch.save(model2.state_dict(), 'pretrained_resnet_model.pth')