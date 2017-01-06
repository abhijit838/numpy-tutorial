import numpy as np

"""
Created by Abhijit[abhijit.maity2010@gmail.com]
Date: 06-01-17
"""

print('Create array with range 15, shape(3, 5)::: ')
a = np.arange(15).reshape(3, 5)
print(a)

print('Shape of the array::: ', a.shape)

print('Dimension of the array::: ', a.ndim)

print('Data type of the array::: ', a.dtype.name)

print('Item size of the array::: ', a.itemsize)

print('Size of the array::: ', a.size)

print('Type of the array::: ', type(a))

print('Create another array b::: ')
b = np.array([3, 4, 6])
print(b)
print('Type of the new array b::: ', type(b))
