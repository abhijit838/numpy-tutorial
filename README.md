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

