# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 22:23:47 2025

@author: Даниш
"""
import torch
import random

'''Случайное целое значение'''
x = torch.randint(0, 300, (1,))
'''Преобразование к float и включение градиента'''
xfloat = x.to(dtype=torch.float32)
xfloat.requires_grad_(True)
print("Исходный x:", x)
print("xfloat с градиентом:", xfloat)

'''мой номер в группе = 6'''
n = 3

'''Возведение в степень'''
power_result = xfloat ** n
print("x^3 =", power_result)

'''Умножение на рандомное число'''
s = random.randint(1, 10)
g = xfloat * s
print("g = x *", s, "=", g)

'''Взятие экспоненты'''
e = torch.exp(g)
print("exp(g) =", e)

'''Создание функции'''
f = xfloat ** n

'''Вычисляем градиент'''
f.backward()
print(f'Производная df/dx для f(x) = x^3: {xfloat.grad}')

'''Проверка аналитически'''
print(f'Аналитическая проверка 3*x^2 = {3 * (xfloat.item() ** 2)}')