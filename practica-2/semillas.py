import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.ClassPerceptron import Perceptron

df = pd.read_csv('./dataFiles/Semillas.csv', encoding='ISO-8859-1')


# variables
X = []
for i in df.iloc:
    # Array de filas desde Tiene_Pelo hasta Tamano_Gato
    X.append(i[:7].values)
X = np.array(X)
# print(X)
T = df['Clase'].replace("Tipo2", 1).replace(
    ["Tipo1", "Tipo3"], 0).astype(int).values
# print(T)

# Entrenar Perceptron
print("Entrenando perceptron...")
ppn = Perceptron(alpha=0.05, n_iter=200, draw=0, title=["Area", "Perimetro",
                                                        "Compacidad", "LongNucleo", "AnchoNucleo", "Asimetria", "LongSurco"])
# fit
X = X.astype(np.float64)
T = T.astype(np.float64)
ppn.fit(X, T)

# Predicción
print("-Predicción-")
Y = ppn.predict(X)
aciertos = sum(Y == T)
print("Aciertos: ", aciertos)
print("%% de aciertos = %.2f %%" % (100*aciertos/X.shape[0]))

print("PESOS: ", ppn.w_)
