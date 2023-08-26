import numpy as np

array = np.array([1, 2, 3, 4])
print(array)

matriz = np.array([(1, 2, 3, 4), (5, 6, 7, 8)])
print(matriz)

# reshape
array = np.arange(2, 3, 0.1)
print(array)

# generate array with zeros
array = np.zeros(5, dtype=int)
print(array)

# array attributes
array = np.array([1, 2, 3, 4])
print(array)
print(array.ndim)
print(array.shape)
print(array.size)
print(array.dtype)

# generate array with random numbers
array = np.random.randint(1, 100, 3)
print(array)

# identity matrix
array = np.identity(3)
print(array)
