# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:34:28 2025

@author: Даниш
"""

import pandas as pd # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt # matplotlib для построения графиков
import numpy as np # numpy для работы с векторами и матрицами
data = pd.read_csv('data.csv', delimiter=',')
#print(data.head())
y = data.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 1, -1)
#print(y)
'''Берем 3 признака'''
X = data.iloc[:, [0, 1, 2]].values
'''Функция нейрона'''
def neuron(w, x):
    inp = w[0] + np.dot(w[1:], x)
    predict = 1 if inp >= 0 else -1
    return predict
'''теперь 4 веса: w0 - w3'''
w = np.random.random(4)
eta = 0.01
wres = []
for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    w[1:] += eta * (target - predict) * xi
    w[0] += eta * (target - predict)
    if j % 10 == 0:
        wres.append(w.tolist())
'''Строим scatter plot для двух признаков (например, длина чашелистика и ширина чашелистика)
А цветом (c) кодируем значение третьего признака (длина лепестка)'''
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], 
                      cmap='viridis', s=80, 
                      edgecolor='k', linewidth=0.7)

'''Разделяем точки по классам с помощью маркеров'''
mask_setosa = (y == 1)
mask_not_setosa = (y == -1)

plt.scatter(X[mask_setosa, 0], X[mask_setosa, 1], 
            c=X[mask_setosa, 2], cmap='viridis', 
            marker='o', s=80, edgecolor='k', linewidth=0.7, label='Сетоса (1)')
plt.scatter(X[mask_not_setosa, 0], X[mask_not_setosa, 1], 
            c=X[mask_not_setosa, 2], cmap='viridis', 
            marker='s', s=80, edgecolor='k', linewidth=0.7, label='Не-Сетоса (-1)')

'''Добавляем цветовую шкалу для третьего признака'''
cbar = plt.colorbar(scatter)
cbar.set_label('Длина лепестка (третий признак)', fontsize=12, fontfamily='DejaVu Sans')

plt.xlabel('Длина чашелистика (первый признак)', fontfamily='DejaVu Sans')
plt.ylabel('Ширина чашелистика (второй признак)', fontfamily='DejaVu Sans')
plt.title('Визуализация 3D-данных в 2D: Цвет = значение третьего признака', 
          fontsize=14, pad=20)
plt.legend()
plt.grid(True, alpha=0.3)

'''Добавляем аннотацию для объяснения'''
plt.annotate('Светлый цвет = маленькое значение\nТемный цвет = большое значение', 
             xy=(0.02, 0.98), xycoords='axes fraction',
             fontsize=10, fontfamily='DejaVu Sans',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
             ha='left', va='top')

plt.tight_layout()
plt.show()