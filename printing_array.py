import numpy as np

print('------------array and reshape----------------------------')

print('1d array::: \n', np.arange(6))

print('2d array::: \n', np.arange(15).reshape(5, 3))

print('3d array::: \n', np.arange(30).reshape(2, 3, 5))

print('-------------------skips elements with ... if it is too long--------------------\n')

print('A very long 1d array::: \n', np.arange(10000))

print('A very long 2d array::: \n', np.arange(10000).reshape(100, 100))

print('---------------do not skip elements with ... -----------------------')

np.set_printoptions(threshold=10000)

print('A very long 2d array::: \n', np.arange(10000).reshape(100, 100))

