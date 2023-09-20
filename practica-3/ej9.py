import pandas as pd
from utils.grafica import *
from utils.ClassNeuronaLineal import NeuronaLineal
from sklearn import preprocessing
from utils.my_functions import Functions as fun

df = pd.read_csv('./dataFiles/automobile-simple.csv')


# # attributes is an array of arrays with the values of each attribute
attributes = df[['curb-weight', 'engine-size', 'horsepower',
                 'city-mpg', 'highway-mpg', 'volume', 'eco-rating', 'price']]

print(attributes.corr())

# price = df['price'].fillna(df['price'].mean())


# X = np.array(attributes)
# Y = np.array(price)


# x_train = X.reshape(-1, 7)
# y_train = Y.reshape(-1, 1)

# # print(x_train, y_train)


# neurona = NeuronaLineal(alpha=0.001, n_iter=300, cotaE=10e-6,
#                         draw=0)

# normalizador = preprocessing.MinMaxScaler()
# x_train = normalizador.fit_transform(x_train)
# y_train = normalizador.fit_transform(y_train)
# print('-- X e Y normalizados linealmente --')

# suma_w = [0, 0, 0, 0, 0, 0, 0]

# # print(suma_w)

# for i in range(0, 50):
#     neurona.fit(x_train, y_train)
#     print('w_: ', neurona.w_)
#     # suma is a list with the sum of each w
#     suma_w = np.add(suma_w, neurona.w_)

# print(suma_w)
# # print('w_: ', neurona.w_)
# # print('b_: ', neurona.b_)

# # print('ECM de mi neurona NORMALIZADA LINEALMENTE CON MI AMIGO SKLEARN: ', fun.ECM_1(
# #     x_train2, y_train2, neurona.w_, neurona.b_), '\n')
