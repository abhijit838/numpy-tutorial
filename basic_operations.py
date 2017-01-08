import numpy as np

"""
Created by Abhijit[abhijit.maity2010@gmail.com]
Date: 07-01-17
"""

print('----------- all arithmetic operations are element wise-------------')
print('------------- 1d array operations---------------')
a = np.array([2, 4, 6, 9])
b = np.arange(4)

c = a * 4
print(a, '*', 4, '=', c)
c = a - b
print(a, '-', b, '=', c)
c = a + b
print(a, '+', b, '=', c)
c = a * b
print(a, '*', b, '=', c)
c = a / b
print(a, '/', b, '=', c)

print(a, '**3 = ', a**3)

print('10 * np.sin(', a, ') = ', 10 * np.sin(a))

print(a)
a *= 3
print('\n*=\n', 3, '\n=\n', a)

print(a, ' < 2 = ', a < 2)

print('---------------2d array operations---same as 1d - element wise-----------')

a = np.arange(4).reshape(2,2)
b = np.array([[1, 5],
              [12, 2]])

print(a, '\n+\n', b, '\n=\n', a + b)

print('--------------------------------matrix dot operations----------------------------------')

print(a, '\ndot\n', b, '\n=\n', np.dot(a, b), '\n')
print(a, '\ndot\n', b, '\n=\n', a.dot(b))

print('----------------- += operation auto type type upcast and exception ---------------------')
a = np.ones((2, 3), dtype='int')
b = np.random.random((2, 3))
print('Type of \n', a, '\nis', a.dtype)
print('Type of \n', b, '\nis', b.dtype)

print('And now\n', b, '\n+=\n', a, '\n=')
b += a
print(b)

print('But\n', a, '\n+=\n', b, '\n=')
# a += b # Uncomment and see the exception
print('So here operations also follows upcasting...')

a = np.ones((2, 3), dtype=np.float64)
print('Now change a dtype to folat64:::\n', a, 'dtype=', a.dtype)

print('Now\n', a, '\n+=\n', b, '\n=')
a += b
print(a)

print('------------------ Normal operation auto upcast to float--------------------')
from numpy import pi

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)

c = a + b
print(a, 'dtype=', a.dtype, '\n+\n', b, 'dtype=', b.dtype, '\n=\n', c, 'dtype=', c.dtype)

print('------------------ Normal operation auto upcast to complex--------------------')
d = np.exp(c * 1j)

print(c, '(dtype=', c.dtype, ')*', '1j=', d, '(dtype=', d.dtype, ')')

print('--------------------Unary operations-----------------------')
a = np.arange(5)
print(a, '.sum() = ', a.sum())
print(a, '.min() = ', a.min())
print(a, '.max() = ', a.max())

print('--------------------Unary operations with axis----------------')
a = np.arange(15).reshape((3, 5))
print(a, '.sum(axis=0) = ', a.sum(axis=0))
print(a, '.max(axis=0) = ', a.max(axis=1))
print(a, '.cumsum(axis=0) = \n', a.cumsum(axis=0))