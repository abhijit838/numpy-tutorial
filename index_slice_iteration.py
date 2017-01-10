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