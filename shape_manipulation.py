import numpy as np

print('************ Shape Manipulation *************')

a = np.floor(10 * np.random.random((3, 4)))
print('Shape of \n', a, ' is = ', a.shape)
"""
Shape of
 [[ 4.  0.  4.  3.]
 [ 1.  8.  7.  2.]
 [ 8.  6.  8.  8.]]  is =  (3, 4)
"""
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

print('-----------Stacking together different arrays-----------')
a = np.floor(10 * np.random.random((3, 4)))
b = np.floor(10 * np.random.random((3, 4)))

print('np.vstack(\n', a, ',\n', b, '\n) = \n', np.vstack((a, b)))

print('np.hstack(\n', a, ',\n', b, '\n) = \n', np.hstack((a, b)))

# The function column_stacks stacks 1D arrays as columns into a 2D array
# Equivalent to vstack for 1D array
# Equivalent to hstack for 2 or more D array
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
print('Create array using r_and with : :::', np.r_[1:4, 0, 4])
# Create array using r_ - c_ and with : ::: [1 2 3 0 4]
print('Create array using c_ and with : :::\n', np.c_[1:4])
"""
Create array using c_ and with : :::
 [[1]
 [2]
 [3]]
"""