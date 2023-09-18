import pandas as pd
import numpy as np
from utils.ClassNeuronaLineal import NeuronaLineal
from utils.my_functions import Functions as fun

df = pd.read_csv('./dataFiles/binario.csv')

bits = df[['b0', 'b1', 'b2']]
decimales = df['dec']

X = np.array(bits)
Y = np.array(decimales)

x_train = X.reshape(-1, 3)
y_train = Y.reshape(-1, 1)

neurona = NeuronaLineal(alpha=0.01, n_iter=300,
                        cotaE=1e-5, draw=0, random_state=None)
neurona.fit(x_train, y_train)
neurona.b_ = 1
# print(neurona.errors_.count())
print('w_: ', neurona.w_)
print('b_: ', neurona.b_)
print(neurona.predict([1, 0, 1]))

print(fun.ECM_2(x_train, y_train, neurona.w_, neurona.b_))
