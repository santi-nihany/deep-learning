import numpy as np


class Functions:
    def ECM_1(X, Y, w, b):
        suma = 0
        dif = 0
        for i in range(0, len(X)):
            pred = w*X[i] + b
            dif = Y[i] - pred
            suma += np.power(dif, 2)
        return suma/len(X)

    def ECM_2(X, Y, w, b):
        suma = 0
        dif = 0
        for i in range(0, len(X)):
            pred = np.dot(w, X[i]) + b
            dif = Y[i] - pred
            suma += np.power(dif, 2)
        return suma/len(X)
