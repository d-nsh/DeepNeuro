# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 10:59:06 2025

@author: Даниш
"""

import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
'''я 6 в списке группы, то есть мне нужно предсказать доход по возрасту'''
data = pd.read_csv('dataset_simple.csv', delimiter=',')
#print(data)
X = data['age'].values
Y = data['income'].values
print(X)
print(Y)
plt.figure(figsize=(10, 8))
plt.scatter(X, Y, alpha = 0.5)
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Зависимость дохода от возраста')
plt.show()
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_norm = scaler_X.fit_transform(X.reshape(-1, 1))
Y_norm = scaler_Y.fit_transform(Y.reshape(-1, 1))
X_tensor = torch.FloatTensor(X_norm)
Y_tensor = torch.FloatTensor(Y_norm)
weight = torch.randn(1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
#print(f'Нач.вес: {weight.item():.2f}')
#print(f'Нач.смещение: {bias.item():.2f}')
def linear_regression(X, weight, bias):
    return X * weight + bias
def train_model(X, Y, weight, bias, learning_rate = 0.01, epochs = 1000):
    losses = []
    for epoch in range(epochs):
        predictions = linear_regression(X, weight, bias)
        loss = torch.mean((predictions - Y) ** 2)
        loss.backward()
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad
            
            weight.grad.zero_()
            bias.grad.zero_()
        losses.append(loss.item())
         
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    return weight, bias, losses
'''обучение'''
final_weight, final_bias, losses = train_model(
    X_tensor, Y_tensor, weight, bias)
print(f'Финальный вес: {final_weight.item():.2f}')
print(f'Финальное смещение: {final_bias.item():.2f}')
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Эпоха')
plt.ylabel('Ошибка (Loss)')
plt.title('Процесс обучения')
plt.show()

with torch.no_grad():
    predictions_normalized = linear_regression(X_tensor, final_weight, final_bias)
    

    predictions_original = scaler_Y.inverse_transform(predictions_normalized.numpy())
    X_original = scaler_X.inverse_transform(X_tensor.numpy())
    Y_original = scaler_Y.inverse_transform(Y_tensor.numpy())
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_original, Y_original, alpha=0.5, label='Реальные данные')
plt.plot(X_original, predictions_original, color='red', linewidth=2, label='Предсказание')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.legend()
plt.title('Линейная регрессия')

plt.subplot(1, 2, 2)
plt.scatter(Y_original, predictions_original, alpha=0.5)
plt.plot([Y_original.min(), Y_original.max()], [Y_original.min(), Y_original.max()], 'r--')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказания')
plt.title('Предсказания vs Реальные значения')

plt.tight_layout()
plt.show()
with torch.no_grad():
    predictions_normalized = linear_regression(X_tensor, final_weight, final_bias)
    predictions_original = scaler_Y.inverse_transform(predictions_normalized.numpy())
    X_original = scaler_X.inverse_transform(X_tensor.numpy())
    Y_original = scaler_Y.inverse_transform(Y_tensor.numpy())
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_original, Y_original, alpha=0.5, label='Реальные данные')
plt.plot(X_original, predictions_original, color='red', linewidth=2, label='Предсказание')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Линейная регрессия: доход vs возраст')

plt.subplot(1, 2, 2)
plt.scatter(Y_original, predictions_original, alpha=0.5)
plt.plot([Y_original.min(), Y_original.max()], [Y_original.min(), Y_original.max()], 'r--')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказания')
plt.grid(True, alpha=0.3)
plt.title('Предсказания vs Реальные значения')

plt.tight_layout()
plt.show()          
