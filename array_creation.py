import numpy as np

"""
Created by Abhijit[abhijit.maity2010@gmail.com]
Date: 06-01-17
"""

print('Creating an array of integers: np.array([1,2,3,4])')
a = np.array([1,2,3,4])
print(a)

print('Data type of a::: ', a.dtype)

print('Creating an array of floats: np.array([1.1,1.3,.4,2.6])')
b = np.array([1.1,1.3,.4,2.6])
print(b)

print('Data type of b::: ', b.dtype)

print('----------------------------auto transform-----------------------------------------')
print('array transforms sequences of sequences into two-dimensional arrays')

print('Creating an array a: np.array([(1,2,3), (4,5,6)])')
a = np.array([(1.1,2,3), (4,5,6)])
print(a)
print('Also notice dtype is also converted for other elements to ::: ', a.dtype.name)

print('----------------------------explicit dtype--------------------------------------------')
print('Creating an array c with explicit dtype::: np.array( [ [1,2], [3,4] ], dtype=complex )')
c = np.array( [ [1,2], [3,4] ], dtype=complex )
print(c)

print('-------------------array with initial placeholders------------------------------------------------')
print('Creating array with zeros::: \n', np.zeros((3, 4)))

print('Creating array with ones::: \n', np.ones((4,5)))

print('Creating array with integers ones::: \n', np.ones((2, 3, 4), dtype=np.int16) )

print('Creating array with rang 15::: \n', np.arange(15).reshape(3,5))

print('Creating empty array with garbage values::: \n', np.empty((2, 3)))

print('Creating array with unknown values in a range with linspace::: \n', np.linspace(0, 2, 9))

from numpy import pi
print('Evaluating sin(x) with linspace::: \n', np.sin(np.linspace(0, 2*pi, 100)))