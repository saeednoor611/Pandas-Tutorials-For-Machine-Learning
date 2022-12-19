# data input output:
import pandas
import pandas as pd

# read csv file
df = pd.read_csv('data.csv')
print(df)

# write csv file

mydf = pd.to_csv('helo')
print(mydf)

# //////////////////////////////
# reading directly from excel file
# read_excel: location must be same
pd.read_excel('example.xlsx',sheet_name='sheet1')

# /////////////////////////////////////
# write to excel
df.to_excel('example2.xlsx',sheet_name='helo')

# /////////////////////////////////////////////
# now let's work with HTML
# reading html page
data = pd.read_html('link of page')

# /////////////////////////////////////////////////

# let's sql with sql
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df.to_sql('mytable',engine)
#
# # read back
sqldf = pd.read_sql('mytable',con=engine)
