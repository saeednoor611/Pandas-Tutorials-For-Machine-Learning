import pandas as pd
mydata = pd.read_csv('adult.csv')
print(mydata.to_string()) # print all data
print(mydata.head(5)) # print five rows
