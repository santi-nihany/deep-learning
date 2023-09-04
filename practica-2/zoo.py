import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.ClassPerceptron import Perceptron

df = pd.read_csv('./dataFiles/zoo.csv', encoding='ISO-8859-1')
# print(df)

# Atributos : 1 si posee la característica, 0 si no (excepto A1, A18, A14)
# Clase (A18) [7] : mamífero, ave, pez, invertebrado, insecto, reptil y anfibio.

# realice un grafico que visualice la cantidad de ejemplos por cada valor del atributo "Clase"
df['Clase'].value_counts().plot(kind='bar')
# plt.savefig("Clase.png")
show = input("Show fig? (Y/N) ")
if (show == 'Y'):
    print('Close matplot tab to continue...')
    plt.show()


# variables
X = []
for i in df.iloc:
    # Array de filas desde Tiene_Pelo hasta Tamano_Gato
    X.append(i[1:17].values)
X = np.array(X)
# print(X[:1])  # imprimo array con 1er elemento

T = df['Clase'].replace("Mamifero", 1).replace(
    ["Ave", "Pez", "Invertebrado", "Insecto", "Reptil", "Anfibio"], 0).astype(int).values
# print(T)

# Entrenar Perceptron
print("Entrenando perceptron...")
ppn = Perceptron(alpha=0.005, n_iter=50, draw=0, title=["Tiene_Pelo", "Tiene_Plumas", "Nace_de_huevo", "Toma_Leche", "Vuela", "Acuatico",
                 "Depredador", "Dentado", "Vertebrado", "Respira", "Venenoso", "Tiene_Aletas", "Cant_Patas", "Tiene_Cola", "Domestico", "Tamano_Gato"])
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

# Pesos
print('Pesos w: \n', ppn.w_, '\n')

# X nuevo
print("Predicción X = [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 4, 1, 0, 1]")
X_nuevo = [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 4, 1, 0, 1]
Y_nuevo = ppn.predict(X_nuevo)
print("Y Nuevo: ", Y_nuevo)
print('bias: ', ppn.b_)
print('weights: \n', ppn.w_)
print("cant errors: ", len(ppn.errors_))
