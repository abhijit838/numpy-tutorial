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
```python
import numpy as np

print('----- Shape Manipulation ------')

a = np.floor(10 * np.random.random((3, 4)))
print('Shape of \n', a, ' is = ', a.shape)
```
* Changing the shape of an array -
```python
print('---------- Changing the shape of an array ------------')
print('Flatten the array ::: ', a.ravel())
# Flatten the array :::  [ 4.  0.  4.  3.  1.  8.  7.  2.  8.  6.  8.  8.]
a.shape = (6, 2)
print('Reshaped array to (6, 2) using shape assignment ::: \n', a.T)
"""
Reshaped array to (6, 2) using shape assignment :::
 [[ 1.  5.  1.  5.  5.  4.]
 [ 8.  6.  3.  6.  4.  8.]]
"""
a = np.floor(10 * np.random.random((3, 4)))
a.resize((2, 6))
print('Reshaped array to (2, 6) using resize ::: \n', a)
"""
Reshaped array to (2, 6) using resize :::
 [[ 0.  7.  4.  0.  7.  6.]
 [ 0.  0.  9.  9.  0.  8.]]
"""
# If you pass -1 to reshape, the other dimensions are automatically calculated
print('Reshaped array to (3, -1) or (3, 4) using reshape ::: \n', a.reshape(3, -1))
```
* Stacking together different array - <br>
For arrays of with more than two dimensions, hstack stacks along their second axes, vstack stacks along their first axes, and concatenate allows for an optional arguments giving the number of the axis along which the concatenation should happen.
```python
print('-----------Stacking together different arrays-----------')
a = np.floor(10 * np.random.random((3, 4)))
b = np.floor(10 * np.random.random((3, 4)))

print('np.vstack(\n', a, ',\n', b, '\n) = \n', np.vstack((a, b)))

print('np.hstack(\n', a, ',\n', b, '\n) = \n', np.hstack((a, b)))

# The function column_stacks stacks 1D arrays as columns into a 2D array
# Equivalent to vstack for 1D array
# Equivalent to hstack for 2 or more D array
print('np.column_stack(\n', a, ',\n', b, '\n) = \n', np.column_stack((a, b)))

a = np.floor(10 * np.random.random((2, 2)))
b = np.floor(10 * np.random.random((2, 2)))
print('np.column_stack(\n', a, ',\n', b, '\n) = \n', np.column_stack((a, b)))
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print('np.concatenate(\n', a, ',\n', b, ', axis=0\n) = \n', np.concatenate((a, b), axis=0))
"""
np.concatenate(
 [[1 2]
 [3 4]] ,
 [[5 6]]
) =
 [[1 2]
 [3 4]
 [5 6]]
"""
print('np.concatenate(\n', a, ',\n', b, ', axis=1\n) = \n', np.concatenate((a, b.T), axis=1))
"""
np.concatenate(
 [[1 2]
 [3 4]] ,
 [[5 6]] , axis=1
) =
 [[1 2 5]
 [3 4 6]]
"""
```
In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis. They allow the use of range literals (”:”) :
```python
print('Create array using r_and with : :::', np.r_[1:4, 0, 4])
# Create array using r_ - c_ and with : ::: [1 2 3 0 4]
print('Create array using c_ and with : :::\n', np.c_[1:4])
"""
Create array using c_ and with : :::
 [[1]
 [2]
 [3]]
"""
```
* Splitting one array to several small ones -
```python
print('----------------- Splitting to several smaller array --------------')
a = np.floor(10 * np.random.random((2, 12)))
print('Splitting into 3 arrays using hsplit.')
print('np.hsplit(', a, ',', 3, '= \n', np.hsplit(a, 3))
"""
Splitting into 3 arrays using hsplit.
np.hsplit( [[ 6.  2.  8.  6.  9.  1.  1.  2.  0.  3.  1.  9.]
 [ 1.  8.  4.  2.  7.  3.  3.  5.  1.  4.  5.  1.]] , 3 =
 [array([[ 6.,  2.,  8.,  6.],
       [ 1.,  8.,  4.,  2.]]), array([[ 9.,  1.,  1.,  2.],
       [ 7.,  3.,  3.,  5.]]), array([[ 0.,  3.,  1.,  9.],
       [ 1.,  4.,  5.,  1.]])]
"""
print('Split after the 3rd and the 4th column.')
print('np.hsplit(', a, '(3, 4)) = \n', np.hsplit(a, (3, 4)))
"""
Split after the 3rd and the 4th column.
np.hsplit( [[ 8.  4.  4.  3.  2.  0.  1.  9.  1.  8.  2.  3.]
 [ 8.  9.  2.  9.  9.  6.  2.  2.  9.  7.  0.  9.]] (3, 4)) =
 [array([[ 8.,  4.,  4.],
       [ 8.,  9.,  2.]]), array([[ 3.],
       [ 9.]]), array([[ 2.,  0.,  1.,  9.,  1.,  8.,  2.,  3.],
       [ 9.,  6.,  2.,  2.,  9.,  7.,  0.,  9.]])]
"""
```
 
#### $ Copies and Views : 
* No Copy at All -
```python
import numpy as np
# ########## copy #############

# ####### No copy #########
# simple assignments make no copy of array object or of their data
a = np.arange(12)
b = a
print(a, ' is ', b, ' = ', a is b)
# [ 0  1  2  3  4  5  6  7  8  9 10 11]  is  [ 0  1  2  3  4  5  6  7  8  9 10 11]  =  True

b.shape = 3, 4
print('Shape of a after changing the shape of b to (3, 4) :::', a.shape)
# Shape of a after changing the shape of b to (3, 4) ::: (3, 4)

# Python passes mutable object as reference, so function call doesn't makes copy


def func(x):
    return id(x)

print('id of a ::: ', id(a), ', id of a from function ::: ', func(a))
# id of a :::  139921269724704 , id of a from function :::  139921269724704

```
* View or Shallow Copy -
```python
# ############ view or shallow copy #############
# Different array objects can share the same data
# The view method creates a new object that looks for same data
c = a.view()
print('View of \n', a, '\n is \n', c)
"""
View of
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
 is
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
"""
print('But \n', a, '\n is \n', c, '\n = \n', a is c)
"""
But
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
 is
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
 =
 False
"""
print(c, '.base \n is \n', a, '\n = \n', c.base is a)
"""
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]] .base
 is
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
 =
 True
"""
print(c, '.flags.owndata = ', c.flags.owndata)
"""
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]] .flags.owndata =  False
"""
c.shape = 2, 6
print('Shape of \n', a, '\n after changing the shape of \n', c, ' to ', c.shape, '\n = \n', a.shape)
"""
Shape of
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
 after changing the shape of
 [[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]]  to  (2, 6)
 =
 (3, 4)
"""
# Slicing an array returns view of it
s = a[:, 1:3]
print('a = ', a, '\n and slice of a, s = \n', s)
"""
a =  [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
 and slice of a, s =
 [[ 1  2]
 [ 5  6]
 [ 9 10]]
"""
s[:] = 2
print('a = ', a, '\n after changing s to \n', s)
"""
a =  [[ 0  2  2  3]
 [ 4  2  2  7]
 [ 8  2  2 11]]
 after changing s to
 [[2 2]
 [2 2]
 [2 2]]
"""
```
* Deep Copy -
```python
# ####### Deep copy ########
# Deep copy creates complete new copy with same data and does not share data
d = a.copy()
print(d, '\n is \n ', a, '\n = \n', d is a)
"""
[[ 0  2  2  3]
 [ 4  2  2  7]
 [ 8  2  2 11]]
 is
  [[ 0  2  2  3]
 [ 4  2  2  7]
 [ 8  2  2 11]]
 =
 False
"""
# no sharing of data
print(d, '.base is \n', a, '\n = \n ', d.base is a)
"""
[[ 0  2  2  3]
 [ 4  2  2  7]
 [ 8  2  2 11]] .base is
 [[ 0  2  2  3]
 [ 4  2  2  7]
 [ 8  2  2 11]]
 =
  False
"""
# New assignment of value also not reflects
d[0, 0] = 1000
print('a = ', a, ' after reassigning the d[0, 0] to 1000')
"""
a =  [[ 0  2  2  3]
 [ 4  2  2  7]
 [ 8  2  2 11]]  after reassigning the d[0, 0] to 1000
"""
```
* Function and Method Overview -
    * Array Creation <br>
        1. numpy.arange([start, ]stop, [step, ]dtype=None)
        2. numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0) : order : {‘K’, ‘A’, ‘C’, ‘F’}
        3. numpy.copy(a, order='K') : order : {‘C’, ‘F’, ‘A’, ‘K’}
        4. numpy.empty(shape, dtype=float, order='C') : order : {‘C’, ‘F’}
        5. numpy.empty_like(a, dtype=None, order='K', subok=True) : order : {‘C’, ‘F’, ‘A’, or ‘K’}
        6. numpy.eye(N, M=None, k=0, dtype=<type 'float'>)
        7. numpy.fromfile(file, dtype=float, count=-1, sep='')
        8. numpy.fromfunction(function, shape, **kwargs)
        9. numpy.identity(n, dtype=None)
        10. numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
        11. numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
        12. numpy.mgrid = <numpy.lib.index_tricks.nd_grid object at 0x49e966ac>
        13. numpy.ogrid = <numpy.lib.index_tricks.nd_grid object at 0x49e9670c>
        14. numpy.ones(shape, dtype=None, order='C') : order : {‘C’, ‘F’}
        15. numpy.ones_like(a, dtype=None, order='K', subok=True) : order : {‘C’, ‘F’, ‘A’, or ‘K’}
        16. numpy.zeros(shape, dtype=float, order='C')order : {‘C’, ‘F’}
        17. numpy.zeros_like(a, dtype=None, order='K', subok=True) : order : {‘C’, ‘F’, ‘A’, or ‘K’}
    * Conversion
        1. ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True) - Copy of the array, cast to a specified type.
        2. numpy.atleast_1d(*arys) - Convert inputs to arrays with at least one dimension. Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
        3. numpy.atleast_2d(*arys) - View inputs as arrays with at least two dimensions.
        4. numpy.atleast_3d(*arys) - View inputs as arrays with at least three dimensions.
        5. numpy.mat(data, dtype=None) - Interpret the input as a matrix. Unlike matrix, asmatrix does not make a copy if the input is already a matrix or an ndarray. Equivalent to matrix(data, copy=False).
    * Manipulations
        1. numpy.array_split(ary, indices_or_sections, axis=0) - Split an array into multiple sub-arrays.
        2. numpy.column_stack(tup) - Stack 1-D arrays as columns into a 2-D array.
        3. numpy.concatenate((a1, a2, ...), axis=0) - Join a sequence of arrays along an existing axis.
        4. numpy.diagonal(a, offset=0, axis1=0, axis2=1) - Return specified diagonals.
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
