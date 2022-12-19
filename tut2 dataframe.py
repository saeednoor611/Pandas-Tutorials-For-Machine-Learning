# DataFrame: A bunch of Series
# a bunch of sereis
import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)
# create dataframe:
df = pd.DataFrame(randn(4,4),['A','B','C','D'],
                  ['col1','col2','col3','col4'])
# print(df)
# Grabs columns
print(df['col2'])
# grabs more than one column
print(df[['col1','col2']])
# add new colum to existing data
print(df)
# remove column or  remove row
df['new'] = df['col1']
print(df)
# check shape of dataframe
print(df.shape)
# grab or select rows (two ways)
print(df.loc['A'])
print(df.iloc[0])
# selecting subsets of rows and columns
print(df.loc[['A','B'],['col2','col4']])
print(df.loc[['D'],['col4']])