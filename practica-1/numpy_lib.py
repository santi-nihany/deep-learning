import numpy as np

array = np.array([1, 2, 3, 4])
print(array)
print(array.ndim)
print(array.shape, " array shape")
print(array.size)
print(array.dtype)
print(array.data)
print(array.size, " array size")
print(array.size * array.itemsize, " array size in bytes")

# operations
print("statistics baby: ", array.sum(),
      array.min(),
      array.max(),
      array.mean(),
      array.std(),
      array.var(),
      array.cumsum(),
      array.cumprod(),
      array.prod(),
      array.ptp(),
      )

matrix = np.array([(1, 2, 3, 4), (5, 6, 7, 8)])
print(matrix)
print(matrix.shape, " matrix shape")
print(matrix.size, " matrix size")

# reshape
array = np.arange(2, 3, 0.1)
print(array)

# generate array with zeros
array = np.zeros(5, dtype=int)
print(array)

# generate array with random numbers
array = np.random.randint(1, 100, 10)
print(array)
array = np.random.choice(array, 3)
print(array, " random choice")

# identity matrix
array = np.identity(3, dtype=int)
print(array)

# diagonal matrix
array = np.diag([1, 2, 3])
print(array)
