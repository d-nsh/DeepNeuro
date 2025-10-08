import torch 
import numpy as np
import pandas as pd
import torch.nn as nn

df = pd.read_csv('data.csv', delimiter=',')
X = df.iloc[:, [0, 1, 2, 3]].values
Y = df.iloc[:, 4].values
Y = np.where(Y == 'Iris-setosa', 1, -1)
X_tensor = torch.tensor(X, dtype = torch.float32)
Y_tensor = torch.tensor(Y, dtype = torch.float32)

model = nn.Linear(4, 1)

criter = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.01) #градиентный спуск

out = model(X_tensor)
loss = criter(out, Y_tensor.reshape(-1, 1))
optim.zero_grad()
loss.backward()
optim.step()

print(f'Loss после одного шага: {loss.item():.4f}')

# Проверка точности
with torch.no_grad():
    predict = model(X_tensor)
    predicted_label = torch.where(predict >= 0, 1, -1)
    accuracy = (predicted_label.flatten() == Y_tensor).float().mean()
    print(f'Точность модели: {accuracy.item()*100:.2f}%')

# Веса модели
print("Обученные веса:")
print(f"Веса: {model.weight}") 
print(f"Смещение: {model.bias}")