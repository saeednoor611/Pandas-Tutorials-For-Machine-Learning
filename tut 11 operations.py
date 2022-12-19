import pandas as pd
import numpy as np

df = pd.DataFrame({
    'col1':[1,2,3,4],
    'col2':[444,555,666,444],
    'col3':['abc','def','ghi','xyz']
})
print(df.head())


# Operations:
# 1 Unique()  Method  ( nunique())
# 2 value_counts()
# 3 apply method
# use custome functions
# use bultin len funciton
# use lanbda expression
# 4 removing columns
# remove
# check columns name
# check column index
# 5: sorting and ordering of dataframe
# 6: null values
# # 7 pivot table method in pandas
# /////////////////////////////////////////////////////////////////////
#
# Q1: Find Out Unique Values. ( Three Method)
print(df['col2'].unique())
print(df['col2'].nunique())
print(df['col2'].value_counts())

# //////////////////////////////////////////////////////////////////////


# Q2: Selecting Data
# a: Conditional Selection
print(df['col1']>2)
# b: Conditional Selection
print((df[df['col1'] > 2]) & (df[df['col2'] == 444]))

# /////////////////////////////////////////////////////////////////////////

# 3 apply method
# use custome functions
# # use lanbda expression
def xxx(a):
    return a * 2
print(df['col1'].apply(xxx))
print(df['col1'].apply(lambda x: x * 2))

# ////////////////////////////////////////////////////////////////////////////
# 4 removing columns
# remove
# check columns name
# check column index
print(df.drop('col1',axis=1))
print(df.columns)
print(df.index())
# /////////////////////////////////////////////////////////////

# 5: sorting and ordering of dataframe
print(df.sort_values(by='col3',axis=1))
# //////////////////////////////////////////////////////////////
# 6: null values
print(df.isnull())
# ////////////////////////////////////////////////////////////////
# 7 pivot table method in pandas
data = {'a':['foo','foo','foo','bar','bar','bar'],
        'b':['one','one','two','two','one','one'],
        'c':['x','y','x','y','x','y'],
        'd':[1,3,2,5,4,1]}
df = pd.DataFrame(data)
print(df)
# # print(df.pivot_table(values='d',index=['a','b'],columns=['c']))
print(df.pivot_table(values='d',index=['a','b'],columns='c'))