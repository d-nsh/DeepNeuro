# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 10:59:06 2025

@author: Даниш
"""

import torch 
import torch.nn as nn 
import pandas as pd
import matplotlib.pyplot as plt
'''я 6 в списке группы, то есть мне нужно предсказать доход по возрасту'''
data = pd.read_csv('dataset_simple.csv', delimiter=',')
'''print(data)
print(data.head())'''
plt.figure(figsize = (10, 6))
plt.scatter(data['age'], data['income'], alpha=0.6, color = 'blue')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Зависимость дохода от возраста')
plt.grid(True, alpha = 0.3)
plt.show()
X = data['age'].values.reshape(-1, 1)
Y = data['income'].values.reshape(-1, 1)
'''print(f'Форма данных {X.shape}')
print(f'Форма данных {Y.shape}')
print(X[:10])
print(Y[:10])'''
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
norm_X = scaler_X.fit_transform(X)
norm_Y = scaler_Y.fit_transform(Y)
print(f'До нормализации возраст: минимальный - {X.min()}, максимальный - {X.max()}, среднее - {X.mean()}')
print(f'После нормализации:минимальный - {norm_X.min()}, максимальный - {norm_X.max()}, среднее - {norm_X.mean()}')
class IncomePredict(nn.Module):
    def __init__(self):
        super(IncomePredict, self).__init__()
        self.layer1 = nn.Linear(1, 50)
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25, 1)
    def forward(self, x):
        x = torch.relu(self.layer1(x))  
        x = torch.relu(self.layer2(x))  
        x = self.layer3(x)              
        return x
model = IncomePredict()
print(model)
params = sum(p.numel() for p in model.parameters())
print(params)
print("Количество весов в layer1:", model.layer1.weight.shape[0] * model.layer1.weight.shape[1])
print("Количество смещений в layer1:", model.layer1.bias.shape[0])
X_tensor = torch.FloatTensor(norm_X)
Y_tensor = torch.FloatTensor(norm_Y)
print(f"X_tensor shape: {X_tensor.shape}")
print(f"Y_tensor shape: {Y_tensor.shape}")
crit = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Функция потерь: MSE (Mean Squared Error)")
print("Оптимизатор: Adam")
print("Learning rate: 0.001")
model.eval()
with torch.no_grad():
    sample_pred = model(X_tensor[:5])
    print(f"\nПредсказания ДО обучения (первые 5):")
    print(sample_pred.numpy().flatten())
    print(f"Реальные значения (первые 5):")
    print(Y_tensor[:5].numpy().flatten())
epochs = 1200
losses = []
for epoch in range(epochs):
    model.train()
    pred = model(X_tensor)
    loss = crit(pred, Y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f'Эпоха {epoch:.2f} |Ошибка :{loss.item():.2f} ')
model.eval()
with torch.no_grad():
    predictions_after = model(X_tensor[:5])
print('Предсказания до обучения')
print([ 0.09135436,  0.12156275,  0.05094078,  0.12610921, -0.00204644])
print('Предсказания после обучения')
print(predictions_after.numpy().flatten())
print('Реальные значения')
print(Y_tensor[:5].numpy().flatten())