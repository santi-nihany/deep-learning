import numpy as np
from keras.utils import to_categorical
a = to_categorical([0, 1, 2, 3], num_classes=4)
# num_classes: numero de clases distintas
# print(a)
a = to_categorical([0, 1, 2, 1], num_classes=3)
# print(a)

# ------ #
x = np.array([[[1, 2, 4, 5]]])
# print(x)
y = np.expand_dims(x, axis=3)
# print(y)
# print(y.shape)

# ------ #
x = np.array([1, 2])
# standard deviation
print(x.std())
