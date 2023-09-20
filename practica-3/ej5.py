import pandas as pd
import matplotlib.pyplot as plt
from utils.grafica import *
from utils.ClassNeuronaLineal import NeuronaLineal

mate = [2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 10, 10]
fisica = [1, 3, 2, 4, 4, 4, 6, 4, 6, 7, 9, 10]

X = np.array(mate)
Y = np.array(fisica)

df = pd.DataFrame({'X': X, 'Y': Y})
print(df.corr())

plot = df.plot.scatter(x='X', y='Y')
plot.set_xlabel('Matematicas')
plot.set_ylabel('Fisica')
plot.set_title('Scatter plot')
# plt.show()

x_train = X.reshape(-1, 1)
y_train = Y.reshape(-1, 1)


neurona = NeuronaLineal(alpha=0.01, n_iter=300, cotaE=1e-5,
                        draw=0, title=["Matematicas", "Fisica"])
neurona.fit(x_train, y_train)

print('w =', neurona.w_[0])
print('b =', neurona.b_[0])

# neurona.w_ = 0.79816056
# neurona.b_ = 0.52217544


suma = 0
dif = 0
for i in range(0, len(x_train)):
    pred = neurona.predict(x_train[i])
    print(' x =', x_train[i], 'y =', y_train[i],
          'y_pred =', pred)
    dif = y_train[i] - pred
    suma += np.power(dif, 2)
    print('gradiente estocastico: ', (-2) * x_train[i] * dif)

print('Error cuadratico medio: ', suma/len(x_train))
print('Predict 9: ', neurona.predict(9))
