# merge and join and concatenat
import pandas as pd

df1 = pd.DataFrame({
    'A': [10, 30, 40, 50, 60, 70],
    'B': [80, 90, 100, 110,120, 130],
    'C': [140, 150, 160, 170, 180, 190]},
    index=[0,1,2,3,4,5]
    )
print(df1)

df2 = pd.DataFrame({
    'A': [10, 30, 40, 50, 60, 70],
    'B': [80, 90, 100, 110,120, 130],
    'C': [140, 150, 160, 170, 180, 190]},
    index=[6,7,8,9,10,11]
    )
print(df2)

df3 = pd.DataFrame({
    'A': [10, 30, 40, 50, 60, 70],
    'B': [80, 90, 100, 110,120, 130],
    'C': [140, 150, 160, 170, 180, 190]},
    index=[12,13,14,15,16,17]
    )
# print(df3)
# operations:
# 1 Concatenation:
# by default it takes 0 axis for concatenating rows
print(pd.concat([df1,df2,df3]))

# 1 Concatenation:
#it takes 1 axis for concatenating rows
print(pd.concat([df1,df2,df3],axis=1))





# 2 Merge:
left = pd.DataFrame({'key':['k0','k1','k2','k3'],
                     'A':['a0','a1','a2','a3'],
                     'B':['b0','b1','b2','b3']})
print(left)
right = pd.DataFrame({'key':['k0','k1','k2','k3'],
                     'C':['c0','c1','c2','c3'],
                     'D':['d0','d1','d2','d3']})
print(right)

# operations:
print(pd.merge(left,right,how='inner',on='key'))

# /////////////////////////////////////////////////////////////////

# 3 Join:
df1 = pd.DataFrame({
    'A': [10, 30, 40, 50, 60, 70],
    'B': [80, 90, 100, 110,120, 130],
    'C': [140, 150, 160, 170, 180, 190]},
    index=['a','b','c','d','e','f']
    )

df2 = pd.DataFrame({
    'A': [10, 30, 40, 50, 60, 70],
    'B': [80, 90, 100, 110,120, 130],
    'C': [140, 150, 160, 170, 180, 190]},
    index=['a','b','c','d','e','f']

    )
print(df1.join(df2))


# ////////////////////////////////////////////////
# # in case of merge we can pass a list of colums
left = pd.DataFrame(
     {
         "key1": ["K0", "K0", "K1", "K2"],
         "key2": ["K0", "K1", "K0", "K1"],
         "A": ["A0", "A1", "A2", "A3"],
         "B": ["B0", "B1", "B2", "B3"],
     }
 )


right = pd.DataFrame(
     {
         "key1": ["K0", "K1", "K1", "K2"],
         "key2": ["K0", "K0", "K0", "K0"],
         "C": ["C0", "C1", "C2", "C3"],
         "D": ["D0", "D1", "D2", "D3"],
     } )

result = pd.merge(left, right, on=["key1", "key2"])
print(result)
