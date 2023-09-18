import pandas as pd
import numpy as np
from utils.ClassNeuronaGral import NeuronaGradiente

df = pd.read_csv('./dataFiles/Semillas.csv')

df['Clase'] = df['Clase'].replace(['Tipo1', 'Tipo2', 'Tipo3'], [0, 1, 0])

df = df.sample(frac=1, random_state=34)

filas_60_porcentaje = int(len(df) * 0.6)

df_train = df.iloc[:filas_60_porcentaje]
df_test = df.iloc[filas_60_porcentaje:]

x_train = np.array(df_train.drop(['Clase'], axis=1))
y_train = np.array(df_train['Clase'])

x_train = x_train.reshape(-1, 7)
y_train = y_train.reshape(-1, 1)

x_test = np.array(df_test.drop(['Clase'], axis=1))
y_test = np.array(df_test['Clase'])


neurona = NeuronaGradiente(alpha=0.2, n_iter=300,
                           cotaE=1e-3, draw=0, FUN='tanh')

N = 30
aciertos_total = 0

for i in range(0, N):
    print('Iteracion ', i)
    neurona.fit(x_train, y_train)
    print('w_: ', neurona.w_)
    print('b_: ', neurona.b_)
    aciertos = 0
    for i in range(0, len(x_test)):
        y_pred = neurona.predict(x_test[i])
        if (((y_pred > 0.8) and (y_test[i] == 1)) or ((y_pred < 0.4) and (y_test[i] == 0))):
            aciertos += 1
    print('Aciertos: ', aciertos)
    print('Tasa de aciertos: ', aciertos * 100 / len(x_test), '\n')
    aciertos_total += aciertos
print('Promedio de aciertos total: ', aciertos_total / N)
