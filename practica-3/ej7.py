import pandas as pd
from utils.grafica import *
from utils.ClassNeuronaLineal import NeuronaLineal
from sklearn import preprocessing
from utils.my_functions import Functions as fun

df = pd.read_csv('./dataFiles/automobile-simple.csv')


engineSize = df['engine-size']
price = df['price'].fillna(df['price'].mean())


X = np.array(engineSize)
Y = np.array(price)


x_train = X.reshape(-1, 1)
y_train = Y.reshape(-1, 1)

# print(x_train, y_train)


neurona = NeuronaLineal(alpha=0.01, n_iter=300, cotaE=1e-5,
                        draw=0, title=["EngineSize", "Price"])
# neurona.fit(x_train, y_train)
# print('--Sin normalizar--')
# print('w_: ', neurona.w_)
# print('b_: ', neurona.b_)

normalizador = preprocessing.MinMaxScaler()
x_train2 = normalizador.fit_transform(x_train)
y_train2 = normalizador.fit_transform(y_train)
print('-- X e Y normalizados linealmente --')
neurona.fit(x_train2, y_train2)
print('w_: ', neurona.w_)
print('b_: ', neurona.b_)

print('ECM de mi neurona NORMALIZADA LINEALMENTE CON MI AMIGO SKLEARN: ', fun.ECM(
    x_train2, y_train2, neurona.w_, neurona.b_), '\n')


# Normalizando ambos atributos utilizando media y desvío.
normalizador = preprocessing.StandardScaler()
x_train3 = normalizador.fit_transform(x_train)
y_train3 = normalizador.fit_transform(y_train)
print('-- X e Y normalizados utilizando media y desvío --')
neurona.fit(x_train3, y_train3)
print('w_: ', neurona.w_)
print('b_: ', neurona.b_)


x_train3 = (x_train-x_train.mean())/x_train.std()
y_train3 = (y_train-y_train.mean())/y_train.std()
print('-- X e Y normalizados utilizando media y desvío  CASERITO --')
neurona.fit(x_train3, y_train3)
print('w_: ', neurona.w_)
print('b_: ', neurona.b_)


print('ECM de mi neurona caserita: ', fun.ECM(
    x_train3, y_train3, neurona.w_, neurona.b_))
