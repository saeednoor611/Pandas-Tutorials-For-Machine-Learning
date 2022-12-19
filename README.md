#
 
PANDAS
Introduction
 Pandas is a Python library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data. The name "Pandas" has a reference to both "Panel Data", and "Python Data Analysis" and was created by Wes McKinney in 2008[1]
Pandas is an open-source library that is made mainly for working with relational or labeled data both easily and intuitively. It provides various data structures and operations for manipulating numerical data and time series. This library is built on top of the NumPy library. Pandas is fast and it has high performance & productivity for users. [2]

Why Use Pandas?
Based on statistical theories, Pandas gives the opportunity to analyze big data and make conclusions. If your messy data sets then pandas can make them readable and relevant. You want strong grip in data science pandas is the best platform in data science and machine learning world.
                             Personal reference.









NOTE:
If you're thinking about data science as a career, then it is imperative that one of the first things you do is learn pandas. If you want to jump deeply in machine world then you have to learn python basics and python’s most commonly used libraries such as,

Top  Python Libraries:
•	TensorFlow.
•	Scikit-Learn.
•	NumPy.
•	Kera’s.
•	PyTorch.
•	LightGBM.
•	Eli5.
•	SciPy.

But in this article, we will keep focus on pandas one of the most famous library of python. We will look into pandas’ functions that are used for data analysis and relational data.


Key Features of Pandas
•	Fast and efficient Data Frame object with default and customized indexing.
•	Tools for loading data into in-memory data objects from different file formats.
•	Data alignment and integrated handling of missing data.
•	Reshaping and pivoting of date sets.
•	Label-based slicing, indexing and subletting of large data sets.
•	Columns from a data structure can be deleted or inserted.
•	Group by data for aggregation and transformations.
•	High performance merging and joining of data.
•	Time Series functionality. [3]


Pandas Environment setup
Standard Python distribution doesn't come bundled with Panda’s module. A lightweight alternative is to install NumPy using popular Python package installer, pip.
pip install pandas
If you install Anaconda Python package, Pandas will be installed by default with the following −
Windows
•	Anaconda (from https://www.continuum.io) is a free Python distribution for SciPy stack. It is also available for Linux and Mac.
•	Canopy (https://www.enthought.com/products/canopy/) is available as free as well as commercial distribution with full SciPy stack for Windows, Linux and Mac.
•	Python (x,y) is a free Python distribution with SciPy stack and Spyder IDE for Windows OS. (Downloadable from http://python-xy.github.io/)
Linux
Package managers of respective Linux distributions are used to install one or more packages in SciPy stack.
For Ubuntu Users
sudo apt-get install python-numpy python-scipy python-matplotlibipythonipythonnotebook
python-pandas python-sympy python-nose
For Fedora Users
sudo yum install numpyscipy python-matplotlibipython python-pandas sympy
python-nose atlas-develop

What's Pandas for?
Pandas has so many uses that it might make sense to list the things it can't do instead of what it can do.
This tool is essentially your data’s home. Through pandas, you get acquainted with your data by cleaning, transforming, and analyzing it.
For example, say you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a Data Frame — a table, basically — then let you do things like:
•	Calculate statistics and answer questions about the data, like
o	What's the average, median, max, or min of each column?
o	Does column A correlate with column B?
o	What does the distribution of data in column C look like?
•	Clean the data by doing things like removing missing values and filtering rows or columns by some criteria
•	Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.
•	Store the cleaned, transformed data back into a CSV, other file or database
Before you jump into the modeling or the complex visualizations you need to have a good understanding of the nature of your dataset and pandas is the best avenue through which to do that.  
From the above graph you can conclude that how much popular pandas is for data science and analyzing data sets.[5]


What is a Series?
A Pandas Series is like a column in a table. It is a one-dimensional array holding data of any type.
Example
Create a simple Pandas Series from a list:
import pandas as pd

a = [1, 7, 2]

myvar = pd. Series(a)

print(myvar)

output:
	
0	1
1	7
2	2

So, in this example we have imported pandas as pd and here we created a set by name a in which we have stored 3 items. Having that we have implemented pandas bult in Series that will generate series.



Create Labels
With the index argument, you can name your own labels.
Example
Create your own labels:
import pandas as pd

a = [1, 7, 2]

myvar = pd. Series (a, index = ["x", "y", "z"])

print(myvar)

output:
	
X	1
Y	7
Z	2

So, here we have updated the above example by using labels while using index ( x   y   z ) that will take place as an index.



Pandas Series Practice:
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

What is a Data Frame?
A Pandas Data Frame is a 2-dimensional data structure, like a 2-dimensional array, or a table with rows and columns.
Example
Create a simple Pandas Data Frame:
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd. DataFrame(data)

print(df) [7]
Result

     calories duration
  0       420        50
  1       380        40
  2       390        45
Firstly we have imported pandas as pd then we have created a dataset by the name of data in which we have two key pair with different values. Having that we have taken a variable by the name of  df that will store 3 d array.
DataFrame: Exercise:
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

this was just an overview of DataFrame in pandas. We will see and explore all the functions that has been used in these examples in upcoming pages.




Condition selection in pandas
It is the ability to perform conditions selections on dataset using
Bracket notations.
It is similar to numpy array.

Example:
import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5,4),['a','b','c','d','e'],['col1','col2','col3','col4'])
print(df)
.
Output
  col1      col2      col3      col4
a 2.706850 0.628133 0.907969 0.503826
b 0.651118 -0.319318 -0.848077 0.605965
c -2.018168 0.740122 0.528813 -0.589001
d 0.188695 -0.758872 -0.933237 0.955057
e 0.190794 1.978757 2.605967 0.683509




let’s perform some conditional selection on this dataset.
# check some selections
booldf = df > 0
print(booldf)
# pass to original dataset
print(df[booldf])
# do everything in single line of code
print(df[df>0])
# pass colms
print(df['col1']>0)
print(df[df['col1']>0])
# pass rows
print(df['col4']<0)
print(df[df['col4']<0])
Output;
   col1   col2   col3   col4
a   True   True   True   True
b   True  False  False   True
c  False   True   True  False
d   True  False  False   True
e   True   True   True   True
       col1      col2      col3      col4
a  2.706850  0.628133  0.907969  0.503826
b  0.651118       NaN       NaN  0.605965
c       NaN  0.740122  0.528813       NaN
d  0.188695       NaN       NaN  0.955057
e  0.190794  1.978757  2.605967  0.683509
       col1      col2      col3      col4
a  2.706850  0.628133  0.907969  0.503826
b  0.651118       NaN       NaN  0.605965
c       NaN  0.740122  0.528813       NaN
d  0.188695       NaN       NaN  0.955057
e  0.190794  1.978757  2.605967  0.683509
a     True
b     True
c    False
d     True
e     True
Name: col1, dtype: bool
       col1      col2      col3      col4
a  2.706850  0.628133  0.907969  0.503826
b  0.651118 -0.319318 -0.848077  0.605965
d  0.188695 -0.758872 -0.933237  0.955057
e  0.190794  1.978757  2.605967  0.683509
a    False
b    False
c     True
d    False
e    False
Name: col4, dtype: bool
       col1      col2      col3      col4
c -2.018168  0.740122  0.528813 -0.589001
it will return true where the condition becomes true, in other case it will return false.
One thing you have to remember, when you pass that condition into the entire dataset, then it will return NaN, where value becomes false.

Pass Multiple columns

# pass multiple colms
print(df[df['col1']>0][['col2','col3']])
# or do in multiple lines
boolser = df['col1']>0
result = df[boolser]
mycol = ['col2','col3']

print(result[mycol])



Output:
     col2      col3
a  0.628133  0.907969
b -0.319318 -0.848077
d -0.758872 -0.933237
e 1.978757 2.605967
       col2      col3
a  0.628133  0.907969
b -0.319318 -0.848077
d -0.758872 -0.933237
e 1.978757  2.605967
multiple conditions selections
# multiple conditions (selections)
res = df[(df['col1'] >0) & (df['col2']>1)]
print(res)

col1      col2      col3      col4
e 0.190794 1.978757 2.605967 0.683509

whenever you want to perform multiple conditions on your dataset then you can’t perform python normal (and or) operators, because it will give ambiguity error.
When You’re working on such type of conditions you have to use (& |) these operations.

Set reset index of dataset
# set reset index of dataset
print(df.reset_index())

  index      col1      col2      col3      col4
0     a  2.706850  0.628133  0.907969  0.503826
1     b  0.651118 -0.319318 -0.848077  0.605965
2     c -2.018168  0.740122  0.528813 -0.589001
3     d  0.188695 -0.758872 -0.933237  0.955057
4     e  0.190794  1.978757  2.605967  0.683509

Initially we have index (a, b,c,d,e) but using or reseting indices of dataset it has been converted into a column and reset index()
Function created new indices for us. (0,1,2,3,4,5)

Another way:
# another way
newind = 'aa bb cc dd ff'.split()
df['states'] = newind
print(df)

print(df.set_index('states'))

Output:
       col1      col2      col3      col4 states
a  2.706850  0.628133  0.907969  0.503826     aa
b  0.651118 -0.319318 -0.848077  0.605965     bb
c -2.018168  0.740122  0.528813 -0.589001     cc
d  0.188695 -0.758872 -0.933237  0.955057     dd
e  0.190794  1.978757  2.605967  0.683509     ff
first we set a new column by the name of states with all the values.
            col1      col2      col3      col4
states                                        
aa      2.706850  0.628133  0.907969  0.503826
bb      0.651118 -0.319318 -0.848077  0.605965
cc     -2.018168  0.740122  0.528813 -0.589001
dd      0.188695 -0.758872 -0.933237  0.955057
ff      0.190794  1.978757  2.605967  0.683509

then we used set index() function for siting this states column as an index.

iloc[].

There is another method to select multiple rows and columns in Pandas. You can use iloc []. This method uses the index instead of the columns name. The code below returns the same data frame as above
df. iloc[:,:2]
	A	B
2030-01-31	-0.168655	0.587590
2030-02-28	0.689585	0.998266
2030-03-31	0.767534	-0.940617
2030-04-30	0.557299	0.507350
2030-05-31	-1.547836	1.276558
2030-06-30	0.511551	1.572085





Drop a Column
You can drop columns using pd. drop ()
df. Drop (columns= ['A', 'C'])
	B	D
2030-01-31	0.587590	-0.031827
2030-02-28	0.998266	0.475975
2030-03-31	-0.940617	-0.341532
2030-04-30	0.507350	-0.296035
2030-05-31	1.276558	0.523017
2030-06-30	1.572085	-0.594772


Pd.concate()
df1 = pd. DataFrame ({'name': ['John', 'Smith’, ‘Paul'],
                     'Age': ['25', '30', '50']},
                    index= [0, 1, 2])
df2 = pd. DataFrame ({'name': ['Adam', 'Smith' ],
                     'Age': ['26', '11']},
                    index= [3, 4])  
Finally, you concatenate the two DataFrame
df_concat = pd. concat ([df1, df2]) 
df_concat
	Age	name
0	25	John
1	30	Smith
2	50	Paul
3	26	Adam
4	11	Smith





Drop_duplicates
If a dataset can contain duplicates information use, `drop_duplicates` is an easy to exclude duplicate rows. You can see that `df_concat` has a duplicate observation, `Smith` appears twice in the column `name.`
df_concat. drop_duplicates('name')
	Age	name
0	25	John
1	30	Smith
2	50	Paul
3	26	Adam


Sort values
You can sort value with sort values
df_concat. sort values('Age')
	Age	name
4	11	Smith
0	25	John
3	26	Adam
1	30	Smith
2	50	Paul




Rename: change of index
You can use rename to rename a column in Pandas. The first value is the current column name and the second value is the new column name.
df_concat. Rename (columns={"name": "Surname", "Age": "Age_ppl"})
	Age_ppl	Surname
0	25	John
1	30	Smith
2	50	Paul
3	26	Adam
4	11	Smith

























Multilevel indexing DataFrame:   or
Hierary Indexing in pandas.
It is the ability to create a dataframe with more than one sections.
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

Output:
             A         B
a1   1 -0.268956  1.293959
      2 -0.054419 -0.798996
      3 -0.366689  0.504892
a2   1  0.728226 -0.482003
      2 -1.065097 -0.637789
      3 -1.028644  0.447621
Here a1, a2 are the outer index or labels and 1,2,3 are the inner labels with randome values.

# let's index it one by one
print(df.loc['a2'].loc[2])
# you can give it names
df.index.names = ['Groups','Numbers']
print(df)
# grab specific value
print(df.loc['a1'].loc[3]['B'])
                       A         B
Groups Numbers                    
a1     1       -0.268956  1.293959
       2       -0.054419 -0.798996
       3       -0.366689  0.504892
a2     1        0.728226 -0.482003
       2       -1.065097 -0.637789
       3       -1.028644  0.447621
               A         B
How to grab sections and see the above code.
Groups                    
a1     -0.054419 -0.798996
a2     -1.065097 -0.637789

here groups is a column name for outer index and number is a column name for inner index.

Cross section in pandas (xs)
Abitlity to grab more than one rows from each section in dataset.;
# another option (crosssetion xs)
print(df.xs('a1'))
# # it has the ability to return more than rows from both sections
print(df.xs(2,level='Numbers'))

Numbers                    
1       -0.223087 -0.553498
2       -0.592945  1.100668
3       -1.365484 -0.613812
               A         B
Groups                    
a1     -0.592945  1.100668
a2      0.508562 -0.180301









Missing Data in pandas:
Sometime in our datasets may be there are lot of values will be null. So how to dropout that values in pandas or how to fill new values in place of that null values, we will be working on that.
1 dropna():
Use to drop any nan value in a row or column.
2 fillna ():
Use to fill new values in place of null value

import numpy as np
import pandas as pd
dic = {'A':[1,2,np.nan],'B':[1,np.nan,np.nan],'C':[1,2,3]}
# print(dic)
df = pd.DataFrame(dic)
print(df)
     
A    B  C
0  1.0  1.0  1
1  2.0  NaN  2
2  NaN  NaN  3





# let's drop null value
print(df.dropna(axis=1))
print(df.dropna(thresh=2))
# (for col axis=1 , row axis=0, threshold=2)

   C
0  1
1  2
2  3
If we pass axis=1 while using dropna () then it will drop out null value in a column.
     A    B  C
0  1.0  1.0  1
1  2.0  NaN  2
If we pass threshold then it will dropout two or more than two null values in a row.
# replace or fill missing values
print(df)
# print(df.fillna(value='helo'))
print(df['A'].fillna(df['A'].mean()))
  A    B  C
0  1.0  1.0  1
1  2.0  NaN  2
2  NaN  NaN  3
0    1.0
1    2.0
2    1.5
Name: A, dtype: float64






Pandas Range Data
Pandas have a convenient API to create a range of date. Let’s learn with Python Pandas examples:
•	
•	The first parameter is the starting date
•	The second parameter is the number of periods (optional if the end date is specified)
•	The last parameter is the frequency: day: ‘D,’ month: ‘M’ and year: ‘Y.’











So, the above example demonstrates date format. Here we have used data range method belongs to pandas.


















































Inspecting Data
You can check the head or tail of the dataset with head (), or tail () preceded by the name of the panda’s data frame as shown in the below Pandas example:
Step 1) Create a random sequence with numpy. The sequence has 4 columns and 6 rows
random = np. random. randn (6,4)
Step 2) Then you create a data frame using pandas.
Use dates_m as an index for the data frame. It means each row will be given a “name” or an index, corresponding to a date.
# Create data with date
df = pd. DataFrame (random,
                  index=dates_m,
                  columns=list('ABCD'))

Step 3) Using head function              df. Head (3)
	A	B	C	D
2030-01-31	1.139433	1.318510	-0.181334	1.615822
2030-02-28	-0.081995	-0.063582	0.857751	-0.527374
2030-03-31	-0.519179	0.080984	-1.454334	1.314947

Here we supported head() method that will generate first five tuples from your data set as you can see in the above example.

Finally, you give a name to the 4 columns with the argument columns

Step 4) Using tail function
df. Tail (3)
	A	B	C	D
2030-04-30	-0.685448	-0.011736	0.622172	0.104993
2030-05-31	-0.935888	-0.731787	-0.558729	0.768774
2030-06-30	1.096981	0.949180	-0.196901	-0.471556

In step 4 we have taken tail() methos that has generated last three tuples with specified format.
Step 5) An excellent practice to get a clue about the data is to use describe (). It provides the counts, mean, std, min, max and percentile of the dataset.

	A	B	C	D
count	6.000000	6.000000	6.000000	6.000000
Mean	0.002317	0.256928	-0.151896	0.467601
Std	0.908145	0.746939	0.834664	0.908910
Min	-0.935888	-0.731787	-1.454334	-0.527374
25%	-0.643880	-0.050621	-0.468272	-0.327419
50%	-0.300587	0.034624	-0.189118	0.436883
75%	0.802237	0.732131	0.421296	1.178404
Max	1.139433	1.318510	0.857751	1.615822
				
df. describe ()

having that we have used describe() method that has helped while familiarize with our dataset that how many row and columns this data contain, and it express mean, count, stand deviation, min, and overall percentage as well.

Slice Data
The last point of this Python Pandas tutorial is about how to slice a pandas data frame.
You can use the column name to extract data in a particular column as shown in the below Pandas example:
## Slice
### Using name
df['A']

2030-01-31   -0.168655
2030-02-28    0.689585
2030-03-31    0.767534
2030-04-30    0.557299
2030-05-31   -1.547836
2030-06-30    0.511551
Freq: M, Name: A, dtype: float64
To select multiple columns, you need to use two times the bracket, [[..,..]]

You can slice the rows with:
The code below returns the first three rows
### using a slice for row
df [0:3]
	A	B	C	D
2030-01-31	-0.168655	0.587590	0.572301	-0.031827
2030-02-28	0.689585	0.998266	1.164690	0.475975
2030-03-31	0.767534	-0.940617	0.227255	-0.341532


Col ():
The col function is used to select columns by names. As usual, the values before the coma stand for the rows and after refer to the column. You need to use the brackets to select more than one column.
## Multi col
df.loc [: ['A','B']]
	A	B
2030-01-31	-0.168655	0.587590
2030-02-28	0.689585	0.998266
2030-03-31	0.767534	-0.940617
2030-04-30	0.557299	0.507350
2030-05-31	-1.547836	1.276558
2030-06-30	0.511551	1.572085
























Read CSV Files
A simple way to store big data sets is to use CSV files (comma separated files).
CSV files contains plain text and is a well know format that can be read by everyone including Pandas.
In our examples we will be using a CSV file called 'data.csv'.
Example
Load the CSV into a DataFrame:
import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string()) 
Tip: use to_string () to print the entire DataFrame.
If you have a large DataFrame with many rows, Pandas will only return the first 5 rows, and the last 5 rows:
Example
Print the DataFrame without the to_string () method:
import pandas as pd

df = pd.read_csv('data.csv')

print(df) 
now this will print first and last five rows.
max_rows
The number of rows returned is defined in Pandas option settings.
You can check your system's maximum rows with the pd.options.display.max_rows statement.
Example
Check the number of maximum returned rows:
import pandas as pd

print(pd.options.display.max_rows) 
In my system the number is 60, which means that if the DataFrame contains more than 60 rows, the print(df) statement will return only the headers and the first and last 5 rows.
You can change the maximum rows number with the same statement.
Example
Increase the maximum number of rows to display the entire DataFrame:
import pandas as pd

pd.options.display.max_rows = 9999

df = pd.read_csv('data.csv')

print(df) 

pandas group by:
allows you to group together rows based off of a column and perform some aggregate functions.
Definition and Usage
The group by() method allows you to group your data and execute functions on these groups.
Syntax
dataframe.transform(by, axis, level, as_index, sort, group_keys, observed, dropna)
________________________________________
Parameters
The axis, level, as_index, sort, group_keys, observed, dropna parameters are keyword arguments.
Example
Find the average co2 consumption for each car brand:
import pandas as pd

data = {
  'co2': [95, 90, 99, 104, 105, 94, 99, 104],
  'model': ['Citigo', 'Fabia', 'Fiesta', 'Rapid', 'Focus', 'Mondeo', 'Octavia', 'B-Max'],
  'car': ['Skoda', 'Skoda', 'Ford', 'Skoda', 'Ford', 'Ford', 'Skoda', 'Ford']
}

df = pd.DataFrame(data)

print(df.groupby(["car"]).mean())
visualization:
 
Example:
import pandas as pd
dic = {'company':['google','facebook','MIT'],
       'person':['Noor','John','Raju'],
       'sales':[300,500,1000]}

df = pd.DataFrame(dic)
print(df)
print(df.groupby('sales').mean())
print(df.groupby().sum().loc['facebook'])
print(df.groupby('google').count())
print(df.groupby('company').min())
print(df.groupby('company').max())
print(df.groupby('company').describe())
what you think of the output.


Read JSON ( pandas data files)
Big data sets are often stored, or extracted as JSON.
JSON is plain text, but has the format of an object, and is well known in the world of programming, including Pandas.
In our examples we will be using a JSON file called 'data.json'.
Open data. Son
Tip: use to_string () to print the entire DataFrame.
________________________________________
Dictionary as JSON
JSON = Python Dictionary
JSON objects have the same format as Python dictionaries.
If your JSON code is not in a file, but in a Python Dictionary, you can load it into a DataFrame directly:
.
Example
Load a Python Dictionary into a DataFrame:
import pandas as pd

data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)

print(df) 
Merge, join, concatenate and compare
pandas provides various facilities for easily combining together Series or DataFrame with various kinds of set logic for the indexes and relational algebra functionality in the case of join / merge-type operations.
In addition, pandas also provides utilities to compare two Series or DataFrame and summarize their differences.
Concatenating objects
The concat() function (in the main pandas namespace) does all of the heavy lifting of performing concatenation operations along an axis while performing optional set logic (union or intersection) of the indexes (if any) on the other axes. Note that I say “if any” because there is only a single possible axis of concatenation for Series.
Before diving into all of the details of concat and what it can do, here is a simple example:
>>>
In [1]: df1 = pd. DataFrame(
   ...:     {
   ...:         "A": ["A0", "A1", "A2", "A3"],
   ...:         "B": ["B0", "B1", "B2", "B3"],
   ...:         "C": ["C0", "C1", "C2", "C3"],
   ...:         "D": ["D0", "D1", "D2", "D3"],
   ...:     },
   ...:     index=[0, 1, 2, 3],
   ...: )
   ...: 

In [2]: df2 = pd. DataFrame(
   ...:     {
   ...:         "A": ["A4", "A5", "A6", "A7"],
   ...:         "B": ["B4", "B5", "B6", "B7"],
   ...:         "C": ["C4", "C5", "C6", "C7"],
   ...:         "D": ["D4", "D5", "D6", "D7"],
   ...:     },
   ...:     index=[4, 5, 6, 7],
   ...: )
   ...: 

In [3]: df3 = pd.DataFrame(
   ...:     {
   ...:         "A": ["A8", "A9", "A10", "A11"],
   ...:         "B": ["B8", "B9", "B10", "B11"],
   ...:         "C": ["C8", "C9", "C10", "C11"],
   ...:         "D": ["D8", "D9", "D10", "D11"],
   ...:     },
   ...:     index=[8, 9, 10, 11],
   ...: )
   ...: 

In [4]: frames = [df1, df2, df3]

In [5]: result = pd.concat(frames)

 
Like its sibling function on ndarrays, numpy.concatenate, pandas.concat takes a list or dict of homogeneously-typed objects and concatenates them with some configurable handling of “what to do with the other axes”:
pd.concat(
    objs,
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    copy=True,
)
•	objs : a sequence or mapping of Series or DataFrame objects. If a dict is passed, the sorted keys will be used as the keys argument, unless it is passed, in which case the values will be selected (see below). Any None objects will be dropped silently unless they are all None in which case a ValueError will be raised.
•	axis : {0, 1, …}, default 0. The axis to concatenate along.
•	join : {‘inner’, ‘outer’}, default ‘outer’. How to handle indexes on other axis(es). Outer for union and inner for intersection.
•	ignore_index : boolean, default False. If True, do not use the index values on the concatenation axis. The resulting axis will be labeled 0, …, n - 1. This is useful if you are concatenating objects where the concatenation axis does not have meaningful indexing information. Note the index values on the other axes are still respected in the join.
•	keys : sequence, default None. Construct hierarchical index using the passed keys as the outermost level. If multiple levels passed, should contain tuples.
•	levels : list of sequences, default None. Specific levels (unique values) to use for constructing a MultiIndex. Otherwise they will be inferred from the keys.
•	names: list, default None. Names for the levels in the resulting hierarchical index.
•	verify_integrity: boolean, default False. Check whether the new concatenated axis contains duplicates. This can be very expensive relative to the actual data concatenation.
•	copy: boolean, default True. If False, do not copy data unnecessarily.
Without a little bit of context many of these arguments don’t make much sense. Let’s revisit the above example. Suppose we wanted to associate specific keys with each of the pieces of the chopped up DataFrame. We can do this using the keys argument:
>>>
In [6]: result = pd.concat(frames, keys=["x", "y", "z"])

 
As you can see (if you’ve read the rest of the documentation), the resulting object’s index has a hierarchical index. This means that we can now select out each chunk by key:
result.loc["y"]
Out[7]: 
    A   B   C   D
4  A4  B4  C4  D4
5  A5  B5  C5  D5
6  A6  B6  C6  D6
7  A7  B7  C7  D7
It’s not a stretch to see how this can be very useful. More detail on this functionality below.
Note
It is worth noting that concat() (and therefore append()) makes a full copy of the data, and that constantly reusing this function can create a significant performance hit. If you need to use the operation over several datasets, use a list comprehension.
frames = [ process_your_file(f) for f in files ]
result = pd.concat(frames)
Note
When concatenating DataFrames with named axes, pandas will attempt to preserve these index/column names whenever possible. In the case where all inputs share a common name, this name will be assigned to the result. When the input names do not all agree, the result will be unnamed. The same is true for MultiIndex, but the logic is applied separately on a level-by-level basis.
Set logic on the other axes
When gluing together multiple DataFrames, you have a choice of how to handle the other axes (other than the one being concatenated). This can be done in the following two ways:
•	Take the union of them all, join='outer'. This is the default option as it results in zero information loss.
•	Take the intersection, join='inner'.
Here is an example of each of these methods. First, the default join='outer' behavior:
>>>
In [8]: df4 = pd.DataFrame(
   ...:     {
   ...:         "B": ["B2", "B3", "B6", "B7"],
   ...:         "D": ["D2", "D3", "D6", "D7"],
   ...:         "F": ["F2", "F3", "F6", "F7"],
   ...:     },
   ...:     index=[2, 3, 6, 7],
   ...: )
   ...: 

In [9]: result = pd.concat([df1, df4], axis=1)

 
Here is the same thing with join='inner':
>>>
In [10]: result = pd.concat([df1, df4], axis=1, join="inner")

 
Lastly, suppose we just wanted to reuse the exact index from the original DataFrame:
>>>
In [11]: result = pd.concat([df1, df4], axis=1).reindex(df1.index)
Similarly, we could index before the concatenation:
>>>
In [12]: pd.concat([df1, df4.reindex(df1.index)], axis=1)
Out[12]: 
    A   B   C   D    B    D    F
0  A0  B0  C0  D0  NaN  NaN  NaN
1  A1  B1  C1  D1  NaN  NaN  NaN
2  A2  B2  C2  D2   B2   D2   F2
3  A3  B3  C3  D3   B3   D3   F3

 
Ignoring indexes on the concatenation axis
For DataFrame objects which don’t have a meaningful index, you may wish to append them and ignore the fact that they may have overlapping indexes. To do this, use the ignore_index argument:
>>>
In [13]: result = pd.concat([df1, df4], ignore_index=True, sort=False)

 
Concatenating with mixed ndims
You can concatenate a mix of Series and DataFrame objects. The Series will be transformed to DataFrame with the column name as the name of the Series.
>>>
In [14]: s1 = pd.Series(["X0", "X1", "X2", "X3"], name="X")

In [15]: result = pd.concat([df1, s1], axis=1)

 
Note
Since we’re concatenating a Series to a DataFrame, we could have achieved the same result with DataFrame.assign(). To concatenate an arbitrary number of pandas objects (DataFrame or Series), use concat.
If unnamed Series are passed they will be numbered consecutively.
>>>
In [16]: s2 = pd.Series(["_0", "_1", "_2", "_3"])

In [17]: result = pd.concat([df1, s2, s2, s2], axis=1)

 
Passing ignore_index=True will drop all name references.
>>>
In [18]: result = pd.concat([df1, s1], axis=1, ignore_index=True)

 
More concatenating with group keys
A fairly common use of the keys argument is to override the column names when creating a new DataFrame based on existing Series. Notice how the default behaviour consists on letting the resulting DataFrame inherit the parent Series’ name, when these existed.
>>>
In [19]: s3 = pd.Series([0, 1, 2, 3], name="foo")

In [20]: s4 = pd.Series([0, 1, 2, 3])

In [21]: s5 = pd.Series([0, 1, 4, 5])

In [22]: pd.concat([s3, s4, s5], axis=1)
Out[22]: 
   foo  0  1
0    0  0  0
1    1  1  1
2    2  2  4
3    3  3  5
Through the keys argument we can override the existing column names.
>>>
In [23]: pd.concat([s3, s4, s5], axis=1, keys=["red", "blue", "yellow"])
Out[23]: 
   red  blue  yellow
0    0     0       0
1    1     1       1
2    2     2       4
3    3     3       5
Let’s consider a variation of the very first example presented:
>>>
In [24]: result = pd.concat(frames, keys=["x", "y", "z"])

 
You can also pass a dict to concat in which case the dict keys will be used for the keys argument (unless other keys are specified):
>>>
In [25]: pieces = {"x": df1, "y": df2, "z": df3}

In [26]: result = pd.concat(pieces)

 
>>>
In [27]: result = pd.concat(pieces, keys=["z", "y"])

 
The MultiIndex created has levels that are constructed from the passed keys and the index of the DataFrame pieces:
>>>
In [28]: result.index.levels
Out[28]: FrozenList([['z', 'y'], [4, 5, 6, 7, 8, 9, 10, 11]])
If you wish to specify other levels (as will occasionally be the case), you can do so using the levels argument:
>>>
In [29]: result = pd.concat(
   ....:     pieces, keys=["x", "y", "z"], levels=[["z", "y", "x", "w"]], names=["group_key"]
   ....: )
   ....: 

 
>>>
In [30]: result.index.levels
Out[30]: FrozenList([['z', 'y', 'x', 'w'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
This is fairly esoteric, but it is actually necessary for implementing things like GroupBy where the order of a categorical variable is meaningful.
Appending rows to a DataFrame
If you have a series that you want to append as a single row to a DataFrame, you can convert the row into a DataFrame and use concat
>>>
In [31]: s2 = pd.Series(["X0", "X1", "X2", "X3"], index=["A", "B", "C", "D"])

In [32]: result = pd.concat([df1, s2.to_frame().T], ignore_index=True)

 
You should use ignore_index with this method to instruct DataFrame to discard its index. If you wish to preserve the index, you should construct an appropriately-indexed DataFrame and append or concatenate those objects.
Database-style DataFrame or named Series joining/merging
pandas has full-featured, high performance in-memory join operations idiomatically very similar to relational databases like SQL. These methods perform significantly better (in some cases well over an order of magnitude better) than other open source implementations (like base::merge.data.frame in R). The reason for this is careful algorithmic design and the internal layout of the data in DataFrame.
See the cookbook for some advanced strategies.
Users who are familiar with SQL but new to pandas might be interested in a comparison with SQL.
pandas provides a single function, merge(), as the entry point for all standard database join operations between DataFrame or named Series objects:
pd.merge(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)
•	left: A DataFrame or named Series object.
•	right: Another DataFrame or named Series object.
•	on: Column or index level names to join on. Must be found in both the left and right DataFrame and/or Series objects. If not passed and left_index and right_index are False, the intersection of the columns in the DataFrames and/or Series will be inferred to be the join keys.
•	left_on: Columns or index levels from the left DataFrame or Series to use as keys. Can either be column names, index level names, or arrays with length equal to the length of the DataFrame or Series.
•	right_on: Columns or index levels from the right DataFrame or Series to use as keys. Can either be column names, index level names, or arrays with length equal to the length of the DataFrame or Series.
•	left_index: If True, use the index (row labels) from the left DataFrame or Series as its join key(s). In the case of a DataFrame or Series with a MultiIndex (hierarchical), the number of levels must match the number of join keys from the right DataFrame or Series.
•	right_index: Same usage as left_index for the right DataFrame or Series
•	how: One of 'left', 'right', 'outer', 'inner', 'cross'. Defaults to inner. See below for more detailed description of each method.
•	sort: Sort the result DataFrame by the join keys in lexicographical order. Defaults to True, setting to False will improve performance substantially in many cases.
•	suffixes: A tuple of string suffixes to apply to overlapping columns. Defaults to ('_x', '_y').
•	copy: Always copy data (default True) from the passed DataFrame or named Series objects, even when reindexing is not necessary. Cannot be avoided in many cases but may improve performance / memory usage. The cases where copying can be avoided are somewhat pathological but this option is provided nonetheless.
•	indicator: Add a column to the output DataFrame called _merge with information on the source of each row. _merge is Categorical-type and takes on a value of left_only for observations whose merge key only appears in 'left' DataFrame or Series, right_only for observations whose merge key only appears in 'right' DataFrame or Series, and both if the observation’s merge key is found in both.
•	validate : string, default None. If specified, checks if merge is of specified type.
o	“one_to_one” or “1:1”: checks if merge keys are unique in both left and right datasets.
o	“one_to_many” or “1:m”: checks if merge keys are unique in left dataset.
o	“many_to_one” or “m:1”: checks if merge keys are unique in right dataset.
o	“many_to_many” or “m:m”: allowed, but does not result in checks.
Note
Support for specifying index levels as the on, left_on, and right_on parameters was added in version 0.23.0. Support for merging named Series objects was added in version 0.24.0.
It is worth spending some time understanding the result of the many-to-many join case. In SQL / standard relational algebra, if a key combination appears more than once in both tables, the resulting table will have the Cartesian product of the associated data. Here is a very basic example with one unique key combination:
>>>
In [33]: left = pd.DataFrame(
   ....:     {
   ....:         "key": ["K0", "K1", "K2", "K3"],
   ....:         "A": ["A0", "A1", "A2", "A3"],
   ....:         "B": ["B0", "B1", "B2", "B3"],
   ....:     }
   ....: )
   ....: 

In [34]: right = pd.DataFrame(
   ....:     {
   ....:         "key": ["K0", "K1", "K2", "K3"],
   ....:         "C": ["C0", "C1", "C2", "C3"],
   ....:         "D": ["D0", "D1", "D2", "D3"],
   ....:     }
   ....: )
   ....: 

In [35]: result = pd.merge(left, right, on="key")

 
Here is a more complicated example with multiple join keys. Only the keys appearing in left and right are present (the intersection), since how='inner' by default.
>>>
In [36]: left = pd.DataFrame(
   ....:     {
   ....:         "key1": ["K0", "K0", "K1", "K2"],
   ....:         "key2": ["K0", "K1", "K0", "K1"],
   ....:         "A": ["A0", "A1", "A2", "A3"],
   ....:         "B": ["B0", "B1", "B2", "B3"],
   ....:     }
   ....: )
   ....: 

In [37]: right = pd.DataFrame(
   ....:     {
   ....:         "key1": ["K0", "K1", "K1", "K2"],
   ....:         "key2": ["K0", "K0", "K0", "K0"],
   ....:         "C": ["C0", "C1", "C2", "C3"],
   ....:         "D": ["D0", "D1", "D2", "D3"],
   ....:     }
   ....: )
   ....: 

In [38]: result = pd.merge(left, right, on=["key1", "key2"])

 
The how argument to merge specifies how to determine which keys are to be included in the resulting table. If a key combination does not appear in either the left or right tables, the values in the joined table will be NA. Here is a summary of the how options and their SQL equivalent names:

Merge method	SQL Join Name	Description
left	LEFT OUTER JOIN	Use keys from left frame only
right	RIGHT OUTER JOIN	Use keys from right frame only
outer	FULL OUTER JOIN	Use union of keys from both frames
inner	INNER JOIN	Use intersection of keys from both frames
cross	CROSS JOIN	Create the Cartesian product of rows of both frames
result = pd.merge(left, right, how="left", on=["key1", "key2"])

 
result = pd.merge(left, right, how="right", on=["key1", "key2"])

 
>>>
In [41]: result = pd.merge(left, right, how="outer", on=["key1", "key2"])

 
>>>
In [42]: result = pd.merge(left, right, how="inner", on=["key1", "key2"])

 
>>>
In [43]: result = pd.merge(left, right, how="cross")

 
You can merge a mult-indexed Series and a DataFrame, if the names of the MultiIndex correspond to the columns from the DataFrame. Transform the Series to a DataFrame using Series.reset_index() before merging, as shown in the following example.
>>>
In [44]: df = pd.DataFrame({"Let": ["A", "B", "C"], "Num": [1, 2, 3]})

In [45]: df
Out[45]: 
  Let  Num
0   A    1
1   B    2
2   C    3

In [46]: ser = pd.Series(
   ....:     ["a", "b", "c", "d", "e", "f"],
   ....:     index=pd.MultiIndex.from_arrays(
   ....:         [["A", "B", "C"] * 2, [1, 2, 3, 4, 5, 6]], names=["Let", "Num"]
   ....:     ),
   ....: )
   ....: 

In [47]: ser
Out[47]: 
Let  Num
A    1      a
B    2      b
C    3      c
A    4      d
B    5      e
C    6      f
dtype: object

In [48]: pd.merge(df, ser.reset_index(), on=["Let", "Num"])
Out[48]: 
  Let  Num  0
0   A    1  a
1   B    2  b
2   C    3  c
Here is another example with duplicate join keys in DataFrames:
>>>
In [49]: left = pd.DataFrame({"A": [1, 2], "B": [2, 2]})

In [50]: right = pd.DataFrame({"A": [4, 5, 6], "B": [2, 2, 2]})

In [51]: result = pd.merge(left, right, on="B", how="outer")

 

Joining on index
DataFrame.join() is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame. Here is a very basic example:
>>>
In [79]: left = pd.DataFrame(
   ....:     {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=["K0", "K1", "K2"]
   ....: )
   ....: 

In [80]: right = pd.DataFrame(
   ....:     {"C": ["C0", "C2", "C3"], "D": ["D0", "D2", "D3"]}, index=["K0", "K2", "K3"]
   ....: )
   ....: 

In [81]: result = left.join(right)

 
>>>
In [82]: result = left.join(right, how="outer")

 
The same as above, but with how='inner'.
>>>
In [83]: result = left.join(right, how="inner")

 
The data alignment here is on the indexes (row labels). This same behavior can be achieved using merge plus additional arguments instructing it to use the indexes:
>>>
In [84]: result = pd.merge(left, right, left_index=True, right_index=True, how="outer")

 
>>>
In [85]: result = pd.merge(left, right, left_index=True, right_index=True, how="inner")

 
Joining key columns on an index
join() takes an optional on argument which may be a column or multiple column names, which specifies that the passed DataFrame is to be aligned on that column in the DataFrame. These two function calls are completely equivalent:
left.join(right, on=key_or_keys)
pd.merge(
    left, right, left_on=key_or_keys, right_index=True, how="left", sort=False
)
Obviously you can choose whichever form you find more convenient. For many-to-one joins (where one of the DataFrame’s is already indexed by the join key), using join may be more convenient. Here is a simple example:
>>>
In [86]: left = pd.DataFrame(
   ....:     {
   ....:         "A": ["A0", "A1", "A2", "A3"],
   ....:         "B": ["B0", "B1", "B2", "B3"],
   ....:         "key": ["K0", "K1", "K0", "K1"],
   ....:     }
   ....: )
   ....: 

In [87]: right = pd.DataFrame({"C": ["C0", "C1"], "D": ["D0", "D1"]}, index=["K0", "K1"])

In [88]: result = left.join(right, on="key")

 
>>>
In [89]: result = pd.merge(
   ....:     left, right, left_on="key", right_index=True, how="left", sort=False
   ....: )
   ....: 

 
To join on multiple keys, the passed DataFrame must have a MultiIndex:
>>>
In [90]: left = pd.DataFrame(
   ....:     {
   ....:         "A": ["A0", "A1", "A2", "A3"],
   ....:         "B": ["B0", "B1", "B2", "B3"],
   ....:         "key1": ["K0", "K0", "K1", "K2"],
   ....:         "key2": ["K0", "K1", "K0", "K1"],
   ....:     }
   ....: )
   ....: 

In [91]: index = pd.MultiIndex.from_tuples(
   ....:     [("K0", "K0"), ("K1", "K0"), ("K2", "K0"), ("K2", "K1")]
   ....: )
   ....: 

In [92]: right = pd.DataFrame(
   ....:     {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]}, index=index
   ....: )
   ....: 
Now this can be joined by passing the two key column names:
>>>
In [93]: result = left.join(right, on=["key1", "key2"])

 
The default for DataFrame.join is to perform a left join (essentially a “VLOOKUP” operation, for Excel users), which uses only the keys found in the calling DataFrame. Other join types, for example inner join, can be just as easily performed:
>>>
In [94]: result = left.join(right, on=["key1", "key2"], how="inner")

 
As you can see, this drops any rows where there was no match.
Joining a single Index to a MultiIndex
You can join a singly-indexed DataFrame with a level of a MultiIndexed DataFrame. The level will match on the name of the index of the singly-indexed frame against a level name of the MultiIndexed frame.
>>>
In [95]: left = pd.DataFrame(
   ....:     {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]},
   ....:     index=pd.Index(["K0", "K1", "K2"], name="key"),
   ....: )
   ....: 

In [96]: index = pd.MultiIndex.from_tuples(
   ....:     [("K0", "Y0"), ("K1", "Y1"), ("K2", "Y2"), ("K2", "Y3")],
   ....:     names=["key", "Y"],
   ....: )
   ....: 

In [97]: right = pd.DataFrame(
   ....:     {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]},
   ....:     index=index,
   ....: )
   ....: 

In [98]: result = left.join(right, how="inner")

 
This is equivalent but less verbose and more memory efficient / faster than this.
>>>
In [99]: result = pd.merge(
   ....:     left.reset_index(), right.reset_index(), on=["key"], how="inner"
   ....: ).set_index(["key","Y"])
   ....: 

 
Joining with two MultiIndexes
This is supported in a limited way, provided that the index for the right argument is completely used in the join, and is a subset of the indices in the left argument, as in this example:
>>>
In [100]: leftindex = pd.MultiIndex.from_product(
   .....:     [list("abc"), list("xy"), [1, 2]], names=["abc", "xy", "num"]
   .....: )
   .....: 

In [101]: left = pd.DataFrame({"v1": range(12)}, index=leftindex)

In [102]: left
Out[102]: 
            v1
abc xy num    
a   x  1     0
       2     1
    y  1     2
       2     3
b   x  1     4
       2     5
    y  1     6
       2     7
c   x  1     8
       2     9
    y  1    10
       2    11

In [103]: rightindex = pd.MultiIndex.from_product(
   .....:     [list("abc"), list("xy")], names=["abc", "xy"]
   .....: )
   .....: 

In [104]: right = pd.DataFrame({"v2": [100 * i for i in range(1, 7)]}, index=rightindex)

In [105]: right
Out[105]: 
         v2
abc xy     
a   x   100
    y   200
b   x   300
    y   400
c   x   500
    y   600

In [106]: left.join(right, on=["abc", "xy"], how="inner")
Out[106]: 
            v1   v2
abc xy num         
a   x  1     0  100
       2     1  100
    y  1     2  200
       2     3  200
b   x  1     4  300
       2     5  300
    y  1     6  400
       2     7  400
c   x  1     8  500
       2     9  500
    y  1    10  600
       2    11  600
If that condition is not satisfied, a join with two multi-indexes can be done using the following code.
>>>
In [107]: leftindex = pd.MultiIndex.from_tuples(
   .....:     [("K0", "X0"), ("K0", "X1"), ("K1", "X2")], names=["key", "X"]
   .....: )
   .....: 

In [108]: left = pd.DataFrame(
   .....:     {"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=leftindex
   .....: )
   .....: 

In [109]: rightindex = pd.MultiIndex.from_tuples(
   .....:     [("K0", "Y0"), ("K1", "Y1"), ("K2", "Y2"), ("K2", "Y3")], names=["key", "Y"]
   .....: )
   .....: 

In [110]: right = pd.DataFrame(
   .....:     {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]}, index=rightindex
   .....: )
   .....: 

In [111]: result = pd.merge(
   .....:     left.reset_index(), right.reset_index(), on=["key"], how="inner"
   .....: ).set_index(["key", "X", "Y"])
   .....: 

 
Merging on a combination of columns and index levels
Strings passed as the on, left_on, and right_on parameters may refer to either column names or index level names. This enables merging DataFrame instances on a combination of index levels and columns without resetting indexes.
>>>
In [112]: left_index = pd.Index(["K0", "K0", "K1", "K2"], name="key1")

In [113]: left = pd.DataFrame(
   .....:     {
   .....:         "A": ["A0", "A1", "A2", "A3"],
   .....:         "B": ["B0", "B1", "B2", "B3"],
   .....:         "key2": ["K0", "K1", "K0", "K1"],
   .....:     },
   .....:     index=left_index,
   .....: )
   .....: 

In [114]: right_index = pd.Index(["K0", "K1", "K2", "K2"], name="key1")

In [115]: right = pd.DataFrame(
   .....:     {
   .....:         "C": ["C0", "C1", "C2", "C3"],
   .....:         "D": ["D0", "D1", "D2", "D3"],
   .....:         "key2": ["K0", "K0", "K0", "K1"],
   .....:     },
   .....:     index=right_index,
   .....: )
   .....: 

In [116]: result = left.merge(right, on=["key1", "key2"])

 
Note
When DataFrames are merged on a string that matches an index level in both frames, the index level is preserved as an index level in the resulting DataFrame.
Note
When DataFrames are merged using only some of the levels of a MultiIndex, the extra levels will be dropped from the resulting merge. In order to preserve those levels, use reset_index on those level names to move those levels to columns prior to doing the merge.
Note
If a string matches both a column name and an index level name, then a warning is issued and the column takes precedence. This will result in an ambiguity error in a future version.
Overlapping value columns
The merge suffixes argument takes a tuple of list of strings to append to overlapping column names in the input DataFrames to disambiguate the result columns:
>>>
In [117]: left = pd.DataFrame({"k": ["K0", "K1", "K2"], "v": [1, 2, 3]})

In [118]: right = pd.DataFrame({"k": ["K0", "K0", "K3"], "v": [4, 5, 6]})

In [119]: result = pd.merge(left, right, on="k")

 
>>>
In [120]: result = pd.merge(left, right, on="k", suffixes=("_l", "_r"))

 
DataFrame.join() has lsuffix and rsuffix arguments which behave similarly.
>>>
In [121]: left = left.set_index("k")

In [122]: right = right.set_index("k")

In [123]: result = left.join(right, lsuffix="_l", rsuffix="_r")

 
Joining multiple DataFrames
A list or tuple of DataFrames can also be passed to join() to join them together on their indexes.
>>>
In [124]: right2 = pd.DataFrame({"v": [7, 8, 9]}, index=["K1", "K1", "K2"])

In [125]: result = left.join([right, right2])

 
Merging together values within Series or DataFrame columns
Another fairly common situation is to have two like-indexed (or similarly indexed) Series or DataFrame objects and wanting to “patch” values in one object from values for matching indices in the other. Here is an example:
>>>
In [126]: df1 = pd.DataFrame(
   .....:     [[np.nan, 3.0, 5.0], [-4.6, np.nan, np.nan], [np.nan, 7.0, np.nan]]
   .....: )
   .....: 

In [127]: df2 = pd.DataFrame([[-42.6, np.nan, -8.2], [-5.0, 1.6, 4]], index=[1, 2])
For this, use the combine_first() method:
>>>
In [128]: result = df1.combine_first(df2)

 
Note that this method only takes values from the right DataFrame if they are missing in the left DataFrame. A related method, update(), alters non-NA values in place:
>>>
In [129]: df1.update(df2)

 
Timeseries friendly merging
Merging ordered data
A merge_ordered() function allows combining time series and other ordered data. In particular it has an optional fill_method keyword to fill/interpolate missing data:
>>>
In [130]: left = pd.DataFrame(
   .....:     {"k": ["K0", "K1", "K1", "K2"], "lv": [1, 2, 3, 4], "s": ["a", "b", "c", "d"]}
   .....: )
   .....: 

In [131]: right = pd.DataFrame({"k": ["K1", "K2", "K4"], "rv": [1, 2, 3]})

In [132]: pd.merge_ordered(left, right, fill_method="ffill", left_by="s")
Out[132]: 
     k   lv  s   rv
0   K0  1.0  a  NaN
1   K1  1.0  a  1.0
2   K2  1.0  a  2.0
3   K4  1.0  a  3.0
4   K1  2.0  b  1.0
5   K2  2.0  b  2.0
6   K4  2.0  b  3.0
7   K1  3.0  c  1.0
8   K2  3.0  c  2.0
9   K4  3.0  c  3.0
10  K1  NaN  d  1.0
11  K2  4.0  d  2.0
12  K4  4.0  d  3.0
Merging asof
A merge_asof() is similar to an ordered left-join except that we match on nearest key rather than equal keys. For each row in the left DataFrame, we select the last row in the right DataFrame whose on key is less than the left’s key. Both DataFrames must be sorted by the key.
Optionally an asof merge can perform a group-wise merge. This matches the by key equally, in addition to the nearest match on the on key.
For example; we might have trades and quotes and we want to asof merge them.
>>>
In [133]: trades = pd.DataFrame(
   .....:     {
   .....:         "time": pd.to_datetime(
   .....:             [
   .....:                 "20160525 13:30:00.023",
   .....:                 "20160525 13:30:00.038",
   .....:                 "20160525 13:30:00.048",
   .....:                 "20160525 13:30:00.048",
   .....:                 "20160525 13:30:00.048",
   .....:             ]
   .....:         ),
   .....:         "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
   .....:         "price": [51.95, 51.95, 720.77, 720.92, 98.00],
   .....:         "quantity": [75, 155, 100, 100, 100],
   .....:     },
   .....:     columns=["time", "ticker", "price", "quantity"],
   .....: )
   .....: 

In [134]: quotes = pd.DataFrame(
   .....:     {
   .....:         "time": pd.to_datetime(
   .....:             [
   .....:                 "20160525 13:30:00.023",
   .....:                 "20160525 13:30:00.023",
   .....:                 "20160525 13:30:00.030",
   .....:                 "20160525 13:30:00.041",
   .....:                 "20160525 13:30:00.048",
   .....:                 "20160525 13:30:00.049",
   .....:                 "20160525 13:30:00.072",
   .....:                 "20160525 13:30:00.075",
   .....:             ]
   .....:         ),
   .....:         "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
   .....:         "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
   .....:         "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
   .....:     },
   .....:     columns=["time", "ticker", "bid", "ask"],
   .....: )
   .....: 
>>>
In [135]: trades
Out[135]: 
                     time ticker   price  quantity
0 2016-05-25 13:30:00.023   MSFT   51.95        75
1 2016-05-25 13:30:00.038   MSFT   51.95       155
2 2016-05-25 13:30:00.048   GOOG  720.77       100
3 2016-05-25 13:30:00.048   GOOG  720.92       100
4 2016-05-25 13:30:00.048   AAPL   98.00       100

In [136]: quotes
Out[136]: 
                     time ticker     bid     ask
0 2016-05-25 13:30:00.023   GOOG  720.50  720.93
1 2016-05-25 13:30:00.023   MSFT   51.95   51.96
2 2016-05-25 13:30:00.030   MSFT   51.97   51.98
3 2016-05-25 13:30:00.041   MSFT   51.99   52.00
4 2016-05-25 13:30:00.048   GOOG  720.50  720.93
5 2016-05-25 13:30:00.049   AAPL   97.99   98.01
6 2016-05-25 13:30:00.072   GOOG  720.50  720.88
7 2016-05-25 13:30:00.075   MSFT   52.01   52.03
By default we are taking the asof of the quotes.
>>>
In [137]: pd.merge_asof(trades, quotes, on="time", by="ticker")
Out[137]: 
                     time ticker   price  quantity     bid     ask
0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98
2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN
We only asof within 2ms between the quote time and the trade time.
>>>
In [138]: pd.merge_asof(trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("2ms"))
Out[138]: 
                     time ticker   price  quantity     bid     ask
0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
1 2016-05-25 13:30:00.038   MSFT   51.95       155     NaN     NaN
2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN
We only asof within 10ms between the quote time and the trade time and we exclude exact matches on time. Note that though we exclude the exact matches (of the quotes), prior quotes do propagate to that point in time.
>>>
In [139]: pd.merge_asof(
   .....:     trades,
   .....:     quotes,
   .....:     on="time",
   .....:     by="ticker",
   .....:     tolerance=pd.Timedelta("10ms"),
   .....:     allow_exact_matches=False,
   .....: )
   .....: 
Out[139]: 
                     time ticker   price  quantity    bid    ask
0 2016-05-25 13:30:00.023   MSFT   51.95        75    NaN    NaN
1 2016-05-25 13:30:00.038   MSFT   51.95       155  51.97  51.98
2 2016-05-25 13:30:00.048   GOOG  720.77       100    NaN    NaN
3 2016-05-25 13:30:00.048   GOOG  720.92       100    NaN    NaN
4 2016-05-25 13:30:00.048   AAPL   98.00       100    NaN    NaN
Comparing objects
The compare() and compare() methods allow you to compare two DataFrame or Series, respectively, and summarize their differences.
This feature was added in V1.1.0.
For example, you might want to compare two DataFrame and stack their differences side by side.
>>>
In [140]: df = pd.DataFrame(
   .....:     {
   .....:         "col1": ["a", "a", "b", "b", "a"],
   .....:         "col2": [1.0, 2.0, 3.0, np.nan, 5.0],
   .....:         "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
   .....:     },
   .....:     columns=["col1", "col2", "col3"],
   .....: )
   .....: 

In [141]: df
Out[141]: 
  col1  col2  col3
0    a   1.0   1.0
1    a   2.0   2.0
2    b   3.0   3.0
3    b   NaN   4.0
4    a   5.0   5.0
>>>
In [142]: df2 = df.copy()

In [143]: df2.loc[0, "col1"] = "c"

In [144]: df2.loc[2, "col3"] = 4.0

In [145]: df2
Out[145]: 
  col1  col2  col3
0    c   1.0   1.0
1    a   2.0   2.0
2    b   3.0   4.0
3    b   NaN   4.0
4    a   5.0   5.0
>>>
In [146]: df.compare(df2)
Out[146]: 
  col1       col3      
  self other self other
0    a     c  NaN   NaN
2  NaN   NaN  3.0   4.0
By default, if two corresponding values are equal, they will be shown as NaN. Furthermore, if all values in an entire row / column, the row / column will be omitted from the result. The remaining differences will be aligned on columns.
If you wish, you may choose to stack the differences on rows.
>>>
In [147]: df.compare(df2, align axis=0)
Out[147]: 
        col1  col3
0 self     a   NaN
  other    c   NaN
2 self   NaN   3.0
  other  NaN   4.0
If you wish to keep all original rows and columns, set keep_shape argument to True.
>>>
In [148]: df.compare(df2, keep_shape=True)
Out[148]: 
  col1       col2       col3      
  self other self other self other
0    a     c  NaN   NaN  NaN   NaN
1  NaN   NaN  NaN   NaN  NaN   NaN
2  NaN   NaN  NaN   NaN  3.0   4.0
3  NaN   NaN  NaN   NaN  NaN   NaN
4  NaN   NaN  NaN   NaN  NaN   NaN
You may also keep all the original values even if they are equal.
>>>
In [149]: df.compare(df2, keep_shape=True, keep_equal=True)
Out[149]: 
  col1       col2       col3      
  self other self other self other
0    a     c  1.0   1.0  1.0   1.0
1    a     a  2.0   2.0  2.0   2.0
2    b     b  3.0   3.0  3.0   4.0
3    b     b  NaN   NaN  4.0   4.0
4    a     a  5.0   5.0  5.0   5.0


pandas operations

 Operations:
•	 1 Unique()  Method  ( nunique())
 2 value_counts()
 3 apply method
     use custome functions
     use bultin len funciton
     use lanbda expression
 4 removing columns
     remove
     check columns name
     check column index
 5: sorting and ordering of dataframe
 6: null values
  7 pivot table method in pandas



1 Unique ():
Use to find out unique values in your dataset.

Let’s do it by example.
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'col1':[1,2,3,4],
    'col2':[444,555,666,444],
    'col3':['abc','def','ghi','xyz']
})
print(df.head())
   col1  col2 col3
0     1   444  abc
1     2   555  def
2     3   666  ghi
3     4   444  xyz
use we use 3 methods for finding unique values
#
# Q1: Find Out Unique Values. ( Three Method)
print(df['col2'].unique())
print(df['col2'].nunique())
print(df['col2'].value_counts())
[444 555 666]
3
444    2
555    1
666    1




2 selections:
print(df['col1']>2)
# b: Conditional Selection
print((df[df['col1'] > 2]) & (df[df['col2'] == 444]))
0    False
1    False
2     True
3     True

3 apply method
     use custome functions
     use bultin len funciton
     use lanbda expression

# 3 apply method
# use custome functions
# use lanbda expression
def xxx(a):
    return a * 2
print(df['col1'].apply(xxx))
print(df['col1'].apply(lambda x: x * 2))
0    2
1    4
2    6
3    8
Name: col1, dtype: int64
0    2
1    4
2    6
3    8
Name: col1, dtype: int64



4 removing columns
     remove
     check columns name
     check column index
# 4 removing columns
# remove
# check columns name
# check column index
print(df.drop('col1',axis=1))
print(df.columns)
print(df.index())

5 sorting values ()

# 5: sorting and ordering of dataframe
print(df.sort_values(by='col3',axis=1))

6 is null ()
Use to to return either true or false for null values.

# 6: null values
print(df.isnull())
    col1   col2   col3
0  False  False  False
1  False  False  False
2  False  False  False
3  False  False  False
If returns false its mean that there is no null value in the dataframe.




7 pivot table:
# 7 pivot table method in pandas
data = {'a':['foo','foo','foo','bar','bar','bar'],
        'b':['one','one','two','two','one','one'],
        'c':['x','y','x','y','x','y'],
        'd':[1,3,2,5,4,1]}
df = pd.DataFrame(data)
print(df)
# print(df.pivot_table(values='d',index=['a','b'],columns=['c']))
print(df.pivot_table(values='d',index=['a','b'],columns='c'))
Original dataframe    
 a    b  c  d
0  foo  one  x  1
1  foo  one  y  3
2  foo  two  x  2
3  bar  two  y  5
4  bar  one  x  4
5  bar  one  y  1
Pivot table
c          x    y
a   b            
bar one  4.0  1.0
    two  NaN  5.0
foo one  1.0  3.0
    two  2.0  NaN





Data Input Output:
Pandas has the ability to read or write data to avoid a lot of recourse.
For that you need four main resources.
1 CSV
2 Excel
3 HTML
4 SQL
To work with these all resources, you need to install four libraries.
Pip install sqlalchemy
Pip install lxml
Pip install html5lib
Pip install BeautifulSoup4

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

 
