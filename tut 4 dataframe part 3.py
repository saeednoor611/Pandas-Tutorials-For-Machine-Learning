# multi level index  dataframe
# or index hierarchy
import pandas as pd
import numpy as np
from numpy.random import  randn

# build dataset
outside = ['a1','a1','a1','a2','a2','a2']
inside = [1,2,3,1,2,3]
hier_index= list(zip(outside,inside))
hier_index= pd.MultiIndex.from_tuples(hier_index)

# create dataframe
df = pd.DataFrame(randn(6,2),hier_index,['A','B'])
print(df)
# let's index it one by one
print(df.loc['a2'].loc[2])
# you can give it names
df.index.names = ['Groups','Numbers']
print(df)
# grab specific value
print(df.loc['a1'].loc[3]['B'])


# /////////////////////////////////////////////////////
# another option (crosssetion xs)
print(df.xs('a1'))
# # it has the ability to return more than rows from both sections
print(df.xs(2,level='Numbers'))