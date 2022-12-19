import pandas as pd
dic = {'company':['google','facebook','MIT'],
       'person':['Noor','John','Raju'],
       'sales':[300,500,1000]}
df = pd.DataFrame(dic)
print(df)
# Operations

print(df.groupby('sales').mean(numeric_only=True))
print(df.groupby('company').sum().loc['facebook'])
print(df.groupby('person').count())
print(df.groupby('sales').min())
print(df.groupby('sales').max())
# print(df.groupby('company').describe())

