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

#### $ Array creation :
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