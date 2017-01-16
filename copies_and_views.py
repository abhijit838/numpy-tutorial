"""
 Created by plank-abhijit on 16/1/17.
"""
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