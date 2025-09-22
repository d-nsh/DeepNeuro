# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 22:06:26 2025

@author: Даниш
"""
'''импортируем все функции'''
from random import *
'''список с рандомными 10-ю числами'''
s = sample(range(1,101), 10)
'''с помощью генератора списков создадим список с четными числами из списка s'''
ch = [int(x) for x in s if x % 2 == 0]
print(s)
print(sum(ch))

