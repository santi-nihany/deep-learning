import tensorflow as tf
from tensorflow import keras


model = keras.Sequential(
    [
        keras.layers.Dense(2, input_shape=(2,), activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()
