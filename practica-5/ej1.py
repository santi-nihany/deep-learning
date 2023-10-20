import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn import metrics


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# Normalización de datos
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

for i in range(X_train.shape[0]):
    X_train[i] = (X_train[i]-X_train[i].mean())/X_train[i].std()

for i in range(X_test.shape[0]):
    X_test[i] = (X_test[i]-X_test[i].mean())/X_test[i].std()

# --- MODEL DECLARATION --- #

# Params layer Conv2D
F = 4  # -- number of filters --
K = 3  # -- size of kernel ---
S = 1  # -- stride ---
P = 1  # -- padding ---
FUN = "relu"  # -- activation function for layer Conv2D --
PS = 2  # -- size of pooling window  --

model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(filters=F, kernel_size=K, activation=FUN, strides=(S, S)))
# model.add(MaxPooling2D(pool_size=(PS, PS)))
model.add(Flatten())
model.add(Dense(15, activation="tanh"))
model.add(Dense(10, activation="softmax"))
model.summary()

model.compile('SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# --- TRAINING --- #
SLOTS = 128
EPOCHS = 400
es = EarlyStopping(monitor="val_accuracy", patience=3, min_delta=0.001)
H = model.fit(x=X_train, y=Y_train, batch_size=SLOTS, validation_data=(
    X_test, Y_test), epochs=EPOCHS, callbacks=[es])

print("Epocas = %d" % es.stopped_epoch)

accuracy = np.array(H.history["accuracy"])
val_accuracy = np.array(H.history["val_accuracy"])

N = np.arange(0, len(accuracy))
plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, 100*accuracy, label="train_acc")
plt.plot(N, 100*val_accuracy, label="val_acc")
plt.title("Efectividad de Entrenamiento")
plt.xlabel("Época #")
plt.ylabel("Efectividad")
plt.legend(loc="lower right")

plt.show()

# --- EVALUATE --- #
pred = model.evaluate(X_test, Y_test, batch_size=SLOTS)

print("\nEfectividad del modelo con datos de validación: %6.2f%%" %
      (pred[1]*100))


# --- PREDICTION --- #
Y_pred = model.predict(X_test)
Y_pred_nro = np.argmax(Y_pred, axis=1)    # conversión a entero
Y_true = np.argmax(Y_test, axis=1)

report = metrics.classification_report(Y_true, Y_pred_nro)
print("Classification report - Training:\n%s" % report)

cm = metrics.confusion_matrix(Y_true, Y_pred_nro)
print("Confusion matrix:\n%s" % cm)
