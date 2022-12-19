# # pandas series lectures:
# # similar to numpy array.
# # but can access lables .
import numpy as np
import pandas as pd
#
# # my python lists and dictionaries.
labels = ['a','b','c']
my_data = [10,20,30]
arr = np.array(my_data)
dic = {'a':10,'b':20,'c':30}
# #creating Series
myser = pd.Series(dic)
print(myser)
#
# # pandas sereis has a variety of objects
# # 1 holds any type of data
print(pd.Series([10,20,30]))
#
# # let's use index in a Series
#
ser1 = pd.Series([1,2,3,4],['USA','China','Russia','India'])
ser2 = pd.Series([1,3,2,4],['USA','China','India','Russia'])
print(ser1)
print(ser2)
# # now access this series lables or values
print(ser2['China'])

# lets add up these seires
print(ser1 + ser2)

# //////////////////////////////////////////////////////////////////////////
