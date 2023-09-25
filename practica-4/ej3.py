import numpy as np
import pandas as pd

from sklearn import preprocessing, metrics, model_selection
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('./dataFiles/Vinos.csv')


X = np.array(df.iloc[:, 1:])
Y = np.array(df.iloc[:, 0])

# print(X, '\n')
# print(Y, '\n')

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=0.2)

min_max_scaler = preprocessing.StandardScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# print(X_train)
# print(X_test)

mlp = MLPClassifier(activation='logistic', solver='sgd',
                    learning_rate_init=0.05, random_state=1, hidden_layer_sizes=(12, 5, 4), tol=1e-05, verbose=False, max_iter=2000)
mlp.fit(X_train, Y_train)

print('n_iter: ', mlp.n_iter_)
print('loss: ', mlp.loss_)
print('classes: ', mlp.classes_)
print('n_layers: ', mlp.n_layers_)
print('n_outputs: ', mlp.n_outputs_)


print(Y_test)
Y_pred = mlp.predict(X_test)
print(Y_pred)

print(metrics.confusion_matrix(Y_test, Y_pred))
