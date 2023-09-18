import pandas as pd
from utils.grafica import *
from utils.ClassNeuronaLineal import NeuronaLineal

df = pd.read_csv('./dataFiles/CCPP.csv')

AT = df['AT']
PE = df['PE']

X = np.array(AT)
Y = np.array(PE)


neurona = NeuronaLineal(alpha=0.01, n_iter=300, cotaE=1e-5,
                        draw=0, title=["AT", "PE"])

neurona.w_ = -2.16
neurona.b_ = 496.91

suma = 0

for i in range(0, len(X)):
    pred = neurona.predict(X[i])
    dif = Y[i] - pred
    suma += np.power(dif, 2)

print('suma: ', suma)
print('E.C.M: ', suma/len(X))


# winners: b y d
