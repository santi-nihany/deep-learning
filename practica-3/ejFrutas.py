import pandas as pd
import numpy as np
from utils.ClassNeuronaGral import NeuronaGradiente

df = pd.read_csv('./dataFiles/FrutasTrain.csv')

X = df[['Diametro', 'Color']]
Y = df['Clase'].replace(['Naranja', 'Melon'], [0, 1])

X = np.array(X)
Y = np.array(Y)


x_train = X.reshape(-1, 2)
y_train = Y.reshape(-1, 1)

neurona = NeuronaGradiente(alpha=0.01, n_iter=500,
                           cotaE=1e-6, draw=0, FUN='sigmoid')
# neurona.fit(x_train, y_train)

neurona.w_ = [2.3944, -2.3891]
neurona.b_ = -0.06368

cant1 = 0
cant2 = 0
cant3 = 0

for i in range(0, 16):
    pred = neurona.predict(X[i])
    if (pred >= 0.8):
        cant1 += 1
    elif (pred <= -0.8):
        cant2 += 1
    else:
        cant3 += 1
    print(pred)

print('mayor o igual que 0.8 ', cant1)
print('menor o igual que -0.8 ', cant2)
print('indef ', cant3)

print(neurona.predict([17, 90]))
