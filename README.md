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
"""
Created by Abhijit[abhijit.maity2010@gmail.com]
Date: 08-01-17
"""
print('--------------- Few universal inbuilt methods----------------')
b = np.arange(5)
print('Create 1d array using arange::: np.arange(5) = ', b)
print('Exponential of e::: np.exp(', b, ') = ', np.exp(b))
print('Square root::: no.sqrt(', b, ') = ', np.sqrt(b))
c = np.array([1, 3, -4, 5, -2])
print('Add::: np.add(', b, ',', c, ') = ', np.add(b, c))

```

    :=> Other inbuilt methodes:
        all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross,
        cumprod, cumsum, diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where
    
#### $ Indexing, Slicing and Iterating : 
```python
import numpy as np

print('----------------- Array index-slicing-iteration -----------------------')

print('---------------One-dimesional array(list operations are applicable)---------------')
a = np.arange(10) ** 3
print('One domensional array :::', a)
# One domensional array ::: [  0   1   8  27  64 125 216 343 512 729]
print('3rd element using index :::', a[2])
# 3rd element using index ::: 8
print('Slice array from 2-5 :::', a[2:5])
# Slice array from 2-5 ::: [ 8 27 64]
a[:6:2] = -100
print('Replace each 2nd element by -100 from start position 6 ::: ', a)
# Replace each 2nd element by -100 from start position 6 :::  [-100    1 -100   27 -100  125  216  343  512  729]
print('Reverse the array ::: ', a[:: -1])
# Reverse the array :::  [ 729  512  343  216  125 -100   27 -100    1 -100]
print('Iteration on ::: ', a)
for i in a:
    print(i ** (1/3))
"""
Iteration on :::  [-100    1 -100   27 -100  125  216  343  512  729]
nan
1.0
nan
3.0
nan
5.0
6.0
7.0
8.0

"""

print('---------------Two-dimensional array---------------')

def func(x, y):
    return 10 * x + y

b = np.fromfunction(func, (5, 4), dtype=int)
print('Two-dimensional array using function ::: \n', b)
"""
Two-dimensional array using function :::
 [[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]

"""
print('[2, 3] position value from the array ::: ', b[2, 3])
# [2, 3] position value from the array :::  23
print('Each row from second column ::: ', b[0:5, 1])
# Each row from second column :::  [ 1 11 21 31 41]
print('Same can be done in other way ::: ', b[:, 1])
# Same can be done in other way :::  [ 1 11 21 31 41]
print('Each column in 2nd, 3rd row of b ::: \n', b[1:3, ])
"""
Each column in 2nd, 3rd row of b :::
 [[10 11 12 13]
 [20 21 22 23]]
"""
print('Last row ::: ', b[-1]) # Equivalent to b[-1, :]
# Last row :::  [40 41 42 43]
print('Last column ::: ', b[:, -1])
# Last column :::  [ 3 13 23 33 43]
print('Reverse row rows ::: \n', b[:: -1, ])
"""
Reverse row rows :::
 [[40 41 42 43]
 [30 31 32 33]
 [20 21 22 23]
 [10 11 12 13]
 [ 0  1  2  3]]
"""
print('Reverse columns ::: \n', b[:,:: -1])
"""
Reverse columns :::
 [[ 3  2  1  0]
 [13 12 11 10]
 [23 22 21 20]
 [33 32 31 30]
 [43 42 41 40]]
"""
print('Reverse row and column ::: \n', b[::-1, ::-1])
"""
Reverse row and column :::
 [[43 42 41 40]
 [33 32 31 30]
 [23 22 21 20]
 [13 12 11 10]
 [ 3  2  1  0]]
"""

print('Multi-dimension array representation with (.) ------')
"""
The expression within brackets in b[i] is treated as an i followed by as many instances of : as needed to represent the remaining axes.
"""
print('Missing index means (:) ::: ', b[1])
# Missing index means (:) :::  [10 11 12 13]
"""
For multi-dimensional array ':' can be represented by '.'
x[1,2,...] is equivalent to x[1,2,:,:,:],

x[...,3] to x[:,:,:,:,3] and

x[4,...,5,:] to x[4,:,:,5,:].
"""
c = np.arange(60).reshape(3, 4, 5)
print('Shape of \n', c, ' is ', c.shape)

print(c, '\n[1,...] = \n', c[1,...])  # same as c[1, :, :] or c[1]
"""
[1,...] =
 [[20 21 22 23 24]
 [25 26 27 28 29]
 [30 31 32 33 34]
 [35 36 37 38 39]]
"""
print(c, '\n[...,2] = \n', c[...,2])  # same as c[:, :, 2]
"""
[...,2] =
 [[ 2  7 12 17]
 [22 27 32 37]
 [42 47 52 57]]
"""
print(c, '\n[...,1,...] = \n', c[...,1, ...])
"""
[...,1,...] =
 [[ 5  6  7  8  9]
 [25 26 27 28 29]
 [45 46 47 48 49]]
"""

print('Iteration on ::: \n', b)
for row in b:
    print(row)
"""
Iteration on :::
 [[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]

[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]

"""
print('Iteration using flat method on :::', b)
for elem in b.flat:
    print(elem)
"""
Iteration using flat method on ::: [[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]
0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43
"""
```
    :=> See also: Indexing, Indexing (reference), newaxis, ndenumerate, indices

#### $ Shape Manipulation :
* Changing the shape of an array -
* Stacking together different array -
* Splitting one array to several small ones -
 
#### $ Copies and Views : 
* No Copy at All -
* View or Shallow Copy -
* Deep Copy -
* Function and Method Overview -
    * Array Creation
    * Conversion
    * Manipulations
    * Questions
    * Ordering
    * Operations
    * Basic Statistics
    * Basic Linear Algebra
    
#### $ Less Basic :
* Broadcasting rules -

#### $ Fancy indexing and index tricks :
* Indexing with Array of Indices -
* Indexing with boolean array - 
* The ix_() function - 
* Indexing with string - 

#### $ Linear Algebra :
* Simple Array Operations - 
* Tips and atricks -
    * Auto Reshaping -
    * Vector Stacking -
    * Histogram -

<a href='https://docs.scipy.org/doc/numpy-dev/reference/index.html#reference'>Numpy Reference link<a/>
    