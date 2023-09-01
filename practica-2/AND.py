from utils.ClassPerceptron import Perceptron
from utils.grafica import *
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([0, 0, 0, 1])
print(X, "\n", T)
ppn = Perceptron(alpha=0.1, n_iter=30, draw=1, title=['X1', 'X2'])
ppn.fit(X, T)

Y = ppn.predict(X)

aciertos = sum(Y == T)
print("Aciertos: ", aciertos)

nAciertos = sum(Y == T)
print("%% de aciertos = %.2f %%" % (100*nAciertos/X.shape[0]))
