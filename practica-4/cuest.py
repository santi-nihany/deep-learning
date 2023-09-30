import numpy as np
import pandas as pd

from sklearn import preprocessing, metrics, model_selection
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('./dataFiles/Sonar.csv')


X = np.array(df.iloc[:, :-1])
Y = np.array(df.iloc[:, -1].replace(['Rock', 'Mine'], [0, 1]))


# print(X)
print(Y)

min_max_scaler = preprocessing.StandardScaler()
X_train = min_max_scaler.fit_transform(X)

# print(X_train)
# print(X_test)

mlp = MLPClassifier(solver='sgd', learning_rate_init=0.1, hidden_layer_sizes=(
    10,), max_iter=2000, verbose=False,  tol=10e-05, activation='tanh')
mlp.fit(X_train, Y)

print('n_iter: ', mlp.n_iter_)
print('loss: ', mlp.loss_)
print('classes: ', mlp.classes_)
print('n_layers: ', mlp.n_layers_)
print('n_outputs: ', mlp.n_outputs_)

# print(Y)
Y_pred = mlp.predict(X)
# print(Y_pred)

print(metrics.confusion_matrix(Y, Y_pred))

# accuracy
print(metrics.accuracy_score(Y, Y_pred))


# ROC curve
Y_pred_proba = mlp.predict_proba(X)
# print(Y_pred_proba)
fpr, tpr, thresholds = metrics.roc_curve(Y, Y_pred_proba[:, 1], pos_label=1)
print('fpr: ', fpr)
print('tpr: ', tpr)
print('thresholds: ', thresholds)
print('auc: ', metrics.auc(fpr, tpr))
