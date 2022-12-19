# missing data
# 1 dropna() : will drop entire row if there
# a single null value
import numpy as np
import pandas as pd
dic = {'A':[1,2,np.nan],'B':[1,np.nan,np.nan],'C':[1,2,3]}
# print(dic)
df = pd.DataFrame(dic)
print(df)
# let's drop null value
print(df.dropna(axis=1))
print(df.dropna(thresh=2))
# (for col axis=1 , row axis=0, threshold=2)


# replace or fill missing values
print(df)
# print(df.fillna(value='helo'))
print(df['A'].fillna(df['A'].mean()))