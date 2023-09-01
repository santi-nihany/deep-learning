import pandas as pd
import numpy as np
from utils.ClassPerceptron import Perceptron
from sklearn import preprocessing

# Abrir archivo hojas.csv como dataframe
df = pd.read_csv("./dataFiles/hojas.csv", header=None)
df.columns = df.iloc[0]
df = df.drop(0)
# print(df, "\n")

# Definición variables
perimetros = df["Perimetro"].values.astype(float)
areas = df["Area"].values.astype(int)

# Array : [[perimetros[i], areas[i]], ... ]
X = np.array([[perimetros[i], areas[i]]
              for i in range(len(perimetros))])
T = df["Clase"].replace("Hoja", 0).replace(
    "Helecho", 1).astype(int).values
# print(T, "\n")

# Entrenamiento perceptron
print("----------Entrenamiento perceptron------------")
ppn = Perceptron(alpha=0.01, n_iter=300, draw=0, title=['Perimetro', 'Area'])
ppn.fit(X, T)

print("------------------Predicción------------------")
# Predicción
Y = ppn.predict(X)

aciertos = sum(Y == T)
print("Aciertos: ", aciertos)
print("%% de aciertos = %.2f %%" % (100*aciertos/X.shape[0]))


print('bias: ', ppn.b_)
print('weights [P,A]: ', ppn.w_)
print('errores cometidos en cada iteración: ', len(ppn.errors_))

X_nuevo = [[770, 5000]]
Y_nuevo = ppn.predict(X_nuevo)
value = 'Helecho' if Y_nuevo == 1 else 'Hoja'
print('Prediction [770, 5000]: ', Y_nuevo, '=>', value)

print('----------Normalización de datos----------')
xTrain = X
normalizador = preprocessing.MinMaxScaler()
xTrain = normalizador.fit_transform(xTrain)
print(X, '\n')
print(xTrain, '\n')

print('----------Entrenamiento perceptron con datos normalizados----------')
# 50 iteraciones Aleatorio
cantExito = 0
print(X, '\n')
print(T)
for i in range(50):
    ppn = Perceptron(alpha=0.005, n_iter=100, draw=0,
                     title=['Perimetro', 'Area'])
    ppn.fit(xTrain, T)
    if (len(ppn.errors_) < 100):
        cantExito = cantExito+1
print("% de ejecuciones en la que alcanzo el 100% de acierto: ", ((cantExito*100)/50))
