from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics, model_selection


df = pd.read_csv("./dataFiles/Balance.csv")
X = np.array(df.iloc[:, 1:])
Y = np.array(df.iloc[:, 0].replace(["L", "R", "B"], [0, 1, 2]))

Y_one_hot = keras.utils.to_categorical(Y, num_classes=3)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y_one_hot, test_size=0.3)

model = keras.Sequential()
model.add(keras.layers.Dense(3, input_shape=[
          4], activation='tanh'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=300, batch_size=5)

Y_pred = model.predict(X_test)
# Convert probabilities to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Calculate confusion matrix
confusion = metrics.confusion_matrix(np.argmax(Y_test, axis=1), Y_pred_classes)
print("Confusion Matrix:")
print(confusion)

score = model.evaluate(X_test, Y_test)
print('Error :', score[0])
print('Accuracy:', score[1])
