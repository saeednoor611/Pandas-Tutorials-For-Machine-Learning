# condition selection
# ability to perform condition selection.
# using brackets
# similar to numpy
import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)

# create data frame
df = pd.DataFrame(randn(3,4),['a','b','c'],['col1','col2','col3','col4'])
print(df)

# check some selections

# # pass colms
# print(df['col1'] > 0)
# print(df[df['col1']> 0])


# # pass rows
# print(df[df['col1'] < 0])

# pass multiple colms
# print(df[df['col1'] > 0] [['col2','col4']])
# /////////////////////////////////////////////////////

# # or do in multiple lines
# boolser = df['col1'] > 0
# result = df[boolser]
# mycol = ['col2','col3']
# print(result[mycol])

# //////////////////////////////////////////
# multiple conditions (selections)
# res = df[(df['col1'] < 0) & (df['col2'] < 1)]
# print(res)
# # ///////////////////////////////////////////////
# # set reset index of dataset
# print(df.reset_index())
# /////////////////////////////////////////////////
# # another way
newind = 'aaa bbb ccc'.split()
df['state']  = newind
print(df.set_index('state'))