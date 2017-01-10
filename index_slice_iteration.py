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

print('---------------Two-dimesional array---------------')
# Coming soon