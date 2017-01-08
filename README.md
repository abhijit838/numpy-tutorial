# Numpy-Tutorial

#### $ Basics of numpy :
  NumPy's main object is the homogenous multidimentional array. It is a table of elements(usually numbers), all of the same type, indexed by a tuple of positive integers. In NumPy dimensiona are called axes. The number of axes is <b>rank</b>.

#### $ Axes Example :
<pre>
[1, 2, 3] - rank 1 as it has one axis. Length - 3

[[0, 1, 2], - rank 2 as it has 2 dimensions of length 2 & 3
 [1, 3, 4]]
</pre>

#### $ Attributes :
Numpy array class is called ndarray also known as the alias array. numpy.array

* ndarray.ndim : Number of axes/dimensions/rank
* ndarray.shape : Tuple of dimension and length. (dimension, length)
* ndarray.size : Total number of elements in array. dimension * length.
* ndarray.dtype : Type of the elements of the array. Default types are python data types. Additionally NumPy provides types of its own. Ex - numpy.int32, numpy.int16, numpy.int64, numpy.float64 etc.
* ndarray.itemsize : Size in bytes of each elements of array. ndarray.dtype.itemsize
* ndarray.data : Buffer contains the actual elements of the array. Alternative to index to get the element.

#### $ Example1 :
```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3,5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<class 'numpy.ndarray'>
>>> b = np.array([3, 4, 6])
>>> b
array([3, 4, 6])
>>> type(b)
<class 'numpy.ndarray'>
```

#### $ Array Creation :
```python
import numpy as np

print('Creating an array of integers: np.array([1,2,3,4])')
a = np.array([1,2,3,4])
print(a)

print('Data type of a::: ', a.dtype)

print('Creating an array of floats: np.array([1.1,1.3,.4,2.6])')
b = np.array([1.1,1.3,.4,2.6])
print(b)

print('Data type of b::: ', b.dtype)

print('-----------------auto transform-------------------------')
print('array transforms sequences of sequences into two-dimensional arrays')

print('Creating an array a: np.array([(1,2,3), (4,5,6)])')
a = np.array([(1.1,2,3), (4,5,6)])
print(a)
print('Also notice dtype is also converted for other elements to ::: ', a.dtype.name)

print('------------------explicit dtype--------------------')
print('Creating an array c with explicit dtype::: np.array( [ [1,2], [3,4] ], dtype=complex )')
c = np.array( [ [1,2], [3,4] ], dtype=complex )
print(c)

print('-------------array with initial placeholders------------')
print('Creating array with zeros::: \n', np.zeros((3, 4)))

print('Creating array with ones::: \n', np.ones((4,5)))

print('Creating array with integers ones::: \n', np.ones((2, 3, 4), dtype=np.int16) )

print('Creating array with rang 15::: \n', np.arange(15).reshape(3,5))

print('Creating empty array with garbage values::: \n', np.empty((2, 3)))

print('Creating array with unknown values in a range with linspace::: \n', np.linspace(0, 2, 9))

from numpy import pi
print('Evaluating sin(x) with linspace::: \n', np.sin(np.linspace(0, 2*pi, 100)))
```

#### $ Array Printing : 

Numpy displays array in a nested lists.

* The last axis is printed from left to right,
* The second-to-last is printed from top to bottom,
* the rest are also printed from top to bottom, with each slice separated from the next by an empty line.

One dimensional arrays are printed as row.

```python
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
```

#### $ Basic Operations : 
```python
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
```

#### $ Universal Functions :
```python
import numpy as np

print('--------------- Few universal inbuilt methods----------------')
b = np.arange(5)
print('Create 1d array using arange::: np.arange(5) = ', b)
print('Exponential of e::: np.exp(', b, ') = ', np.exp(b))
print('Square root::: no.sqrt(', b, ') = ', np.sqrt(b))
c = np.array([1, 3, -4, 5, -2])
print('Add::: np.add(', b, ',', c, ') = ', np.add(b, c))

```