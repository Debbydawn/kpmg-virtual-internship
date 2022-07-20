# import pip
# import xlrd
# import pip
# pip.main(["install", "openpyxl"])
# #pip install xlrd
# import pandas as pd
# csv_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/TopSellingAlbums.csv'
# df = pd.read_csv(csv_path)
# df.head()
# xlsx_path = ("C:/Users/adedi_tpk1ys1/OneDrive/Documents/data.xlsx")
#
# df = pd.read_excel(xlsx_path)
# print(df)
# print()
# top = ("C:/Users/adedi_tpk1ys1/Downloads/TopSellingAlbums.xlsx","r")
# dfs = top.read_excel(top)
# print(dfs)
# df.head()
# x = df[['Length']]
# x
#
# with open("C:/Users/adedi_tpk1ys1/Downloads/TopSellingAlbums.xlsx","r") as top:
#     dfs = top.read()
#     print(dfs)
#     #dfs.head()

# Reading an excel file using Python
# import xlrd
#
# # Give the location of the file
# loc = ("C:/Users/adedi_tpk1ys1/Downloads/TopSellingAlbums.xlsx")
#
# # To open Workbook
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
#
# # For row 0 and column 0
# print(sheet.cell_value(0, 0))
# import numpy as np
# a = np.array([0,1,2,3,4])
# print(type(a))
# print(a.dtype)
# print(a.size)
# print(a.ndim)
# print(a.shape)
# print()
# b = np.array([3.1,11,.02,6.2,213.2,5.2])
# print(type(b))
# print()
# c = np.array([20,1,2,3,4])
# print(c)
# print(b.mean())
# print(np.linspace(-5,2,num=9))
# print()
# x = np.linspace(0,2*np.pi,100)
# y = np.sin(x)
# import matplotlib.pyplot as plt
# #matplotlib inline
# plt.plot(x,y)
# 'C:/Users/adedi_tpk1ys1/AppData/Local/Programs/Python/Python39/python.exe -m pip install --upgrade pip'

# a=np.array([0,1])
# b=np.array([1,0])
# print(np.dot(a,b))
# X=np.array([[1,0,1],[2,2,2]])
# out=X[0:2,2]
# print(out)
#
# X=np.array([[1,0],[0,1]])
# Y=np.array([[2,1],[1,2]])
# Z=np.dot(X,Y)
# print(Z)
# print(2//3)
# x= 6
# if(x!=1):
#  print('How are you?')
# else:
#  print('Hi')
#
# A=[8,5,2]
# for a in A:
#  print(12-a)
#
# class Rectangle(object):
#  def __init__(self,width=2,height =3,color='r'):
#   self.height=height
#   self.width=width
#   self.color=color
#   def drawRectangle(self):
#    plt.gca().add_patch(plt.Rectangle((0, 0),self.width, self.height ,fc=self.color))
#    plt.axis('scaled')
#    plt.show()
#   print(height)
# var = '01234567'
# print(var[::2])
# import os
# print(os.getcwd())  # to check present directory
# print(os.listdir())  # get list of directory and chdir() to change directory
"""Data science bootcamp"""
# def square_fun(x):
#     square = x**2
#     return square
# print(square_fun(4))
"""using lambda for mathematical operations"""
# square_num = lambda x: x**2
# print(square_num(4))
# sum = lambda x,y,z : x**y+z
# print(sum(3,2,5))
# age = [23,44,17,36,28,21,14,19,15]
# fil = list(filter(lambda x:x<18,age))
# print(fil)
# print()
# list1 = list(range(1000000))
"""numpy dict"""
# import pop as pop
# import inline as inline
# import matplotlib
import numpy as np
import pandas as pd

# np_array = np.arange(1000000)
# #print(list1*3)
# print()
# print()
# #print(np_array*3)
# array = np.random.randn(2,3)
# print(array)
# print(array.ndim)
# print(array.shape)
# list2 = [3,1,5,6,9,3]
# list3 = [[1,2,3],[4,5,6]]
# array2 = np.array(list2)
# print(array2)
# array3 = np.array(list3)
# print(array3)
# print(array2.dtype)
# print(array3.dtype)
# array4 = np.zeros((3,4))
# print(array4)
# print()
# array5 = np.array(list2,dtype = "int16")
# print(array5.dtype)
# arrayz_fk = array2.astype(np.float64)
# print(arrayz_fk)
# list4 = ["19","24","34","14","56"]
# array6 = np.array(list4)
# print(array6.dtype)
# array_age = array6.astype(np.int64)
# print(array_age)
# print(array_age-5)
# slice1 = array2[2:]
# print(slice1)
# slice1[2] = 16
# print(slice1)
# print(array2)
# print()
"""want to make changes to an array without changing the original data"""
# slice2 = array2[2:].copy()
# print(slice2)
# slice2[1] = 34
# print(slice2)
# print(array2)
# print()
"""indexing list of list"""
# array2d = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# print(array2d)
# print(array2d[1:3,1:3])
# print(array2d[2:3,1:2])
# print(array2d[2,2])
"""assigning an array to another array"""
# array1 = np.array(["a","b","c","a","d","b","a"])
# array2 = np.array([22,45,22,61,19,23,31])
# print(array2[array1 == 'b'])
# array3 = np.random.randn(7,3)
# print(array3[array1 == 'a'])
"""indexing a particular set of index"""
# array4 = np.array([0,12,41,2,27,28,23,38,20,6])
# ind =[1,3,5,7]
# print(array4[ind])
# ind2 = np.array([[1,3],[5,7]])
# print(array4[ind2])
# print()
# array2 = np.array([[52,9,30,46],[4,43,84,15],[50,60,52,62]])
# ind3 = ((0,1,2),(3,2,3))
# print(array2[ind3])
"""using separate variables"""
# row = [0,1,2]
# column =[3,2,3]
# print(array2[row,column])
"""transposing array"""
# array1 = np.array ([[2,8,3,19],[6,8,16,13],[13,2,13,6]])
# print(array1.transpose())
"""or"""
# print(array1.T)
# array2 = np.array([1,2,3,4,5,6])
# print(array2.T)
"""Mathematical and statistical methods such as min,max,mean,S.D etc"""
# array3 = np.random.randint(15,size=(4,5))
# print(array3)
# print("mean is: ",array3.mean())
# print("min is: ",array3.min(),"max is: ",array3.max())
# print("std is: ",array3.std())
# print("cumulative sum is: ",array3.cumsum()) # return cumulative sum chart
# print("the index of the min is ",array3.argmin()) # returns the index of the minimum value
# print("the index of the max is ",array3.argmax())
# print("sum across row: ",array3.sum(axis = 0)) # returning sum across a row
# print("sum across col: ",array3.sum(axis = 1)) # returning sum across the columns
# array4 = np.array([True,False,True,False,False])
# print(array4.sum())
# print(array4.any()) # will scan and return true if at least one is true
# print(array4.all()) # will return true if all values are true
"""sorting array"""
# array5 = np.array([3.0,6.0,1,4,9.2,4.8,10.9,5.0])
# print(array5)
# print(array5.sort()) # sort in place i.e changes the original array
# print(np.sort(array5)) # sort into a copy
# print(np.sort(array5)[::-1]) # descending sorting
# print(array3)
# print(array3.sort(1)) # to sort within column in the same row
# print(np.sort(array3))
# print(array3.sort(0)) # to sort within row in the same column
# array6 = np.array(['harry','jake','harry','george','jake','harry','james','jake','jake','harry'])
# print(np.unique(array6)) # return sorted unique values i.e no duplicate
"""save and load numpy array"""
# array7 = np.arange(10)
# print(array7)
# np.save("my_array",array7) # to save a array file
# np.load("my_array.npy") # to load an array file
# array8 = np.load("my_array.npy") # loading and assigning the array
# print(array8)
# np.savez("multiole_array.npz", a1 = array3, a2 = array8, a3 = array5) # saviving multiple array in one file to return a dict when loaded
# array9 = np.load("multiole_array.npz")
# print(array9["a1"])
# print(array9["a3"])

# data1 = pd.Series([12,23,13,44,15])
# print(data1)
# print(data1.values)
# print(data1.index)
# data2 = pd.Series([12,23,13,44,15], index = [1,2,3,4,5])
# print(data2)
# print(data2[[1,3,5]])
# print(data2 < 16) # if you want it to return a bool
# print(data2[data2 <16]) # to return the actual values
# # series can be created from a dict the keys will b the index and the values remain values
# dict ={"sam": 23, "jone": 41, "jake": 26, "sally": 29}
# data3 = pd.Series(dict)
# print(data3)
# print(data3.index, data3.values)
# print("sam" in data3) # to cheek if a particular index is in the series
# data3.index.name = "Names" # to assign a title for the index
# #data3.values.name = "Age"
# print(data3)
# data3.index = ['a','b','c','d']
# print(data3)
# data3.index.name = "Age"
# print(data3)
"""Dataframe in pandas"""  # it can take different data type
# dict1 = {'ind':[1,2,3,4,5,6],'Name':['bob','jake','sam','jone','sally','william'],'Age':[23,34,41,29,19,34], 'Income':[72,65,49,39,81,55]}
# print(dict1)
# dt1 = pd.DataFrame(dict1)
# print(dt1)
# dt1.set_index('ind', inplace = True) # to set a different index
# print(dt1)
# print(dt1.head(3))  # to return the beginning of the dataframe you can specify the number of rows
# print(dt1.tail(2)) # to return thr tail of the dataframe
# print(dt1['Age']) # this works with spaces and all
# print(dt1.Age) # works when it is a valid data name without spaces and all
# print(dt1.loc[2]) # to get a specific variable by using the index to call it
# dt1['Income']= 100 # changing the whole column
# print(dt1)
# dt1['Income'][2] = 200 # chaning a particular index in a column
# print(dt1)
# dt1['Income'] = pd.Series([44,38,79,23,66,59],index = [2,3,1,5,4,6]) # changing the values of a column
# print(dt1)
# Height = [ 5.6,7.1,6.8,5.9,6.1,5.8]
# dt1['Heigh'] = Height
# print(dt1)
# dt1['Height'] = pd.Series(Height,index = [2,3,1,5,4,6])
# print(dt1)
# print(dt1.T)
# print(dt1.columns)
# print(dt1.index)
"""indexing in pandas"""
# labels = list('abcdef')
# series1 = pd.Series([32,45,23,37,65,55], index = labels)
# print(series1)
# dict1 = {'ind':[1,2,3,4,5,6],'Name':['bob','jake','sam','jone','sally','william'],'Age':[23,34,41,29,19,34], 'Income':[72,65,49,39,81,55]}
# dt1 = pd.DataFrame(dict1)
# print(dt1)
# dt1.set_index('ind',inplace = True)
# print(dt1)
# dt1["degree"] = pd.Series(['yes','no','no'], index = [2,4,6])
# print(dt1)
# print(dt1.degree)
# dict2 = {'ind':[1,2,1,4,5,6],'Name':['bob','jake','bob','jone','sally','william'],'Age':[23,34,41,29,19,34], 'Income':[72,65,49,39,81,55]}
# dt2 = pd.DataFrame(dict2)
# dt2.set_index('ind', inplace = True)
# print(dt2)
# print(dt2.loc[1]) # will return all the data the are called(duplicate labels)
# print()
"""reindexing"""
# print(series1)
# series2 = series1.reindex(['b','c','a','e','d','f','g'])
# print(series2)
# series3 = pd.Series(['red','green','yellow'],index =[0,2,5])
# print(series3)
# print(series3.reindex(range(6),method ='ffill')) # ffill means forward filling
# dict3 = {'red': [33, 22, 55], 'green': [66, 33, 11], 'white': [66, 44, 22]}
# dt3 = pd.DataFrame(dict3, index=['a', 'c', 'd'])
# print(dt3)  # reindexing rows and columns
# print(dt3.reindex(['a','b','c','d']))
# print(dt3.reindex(columns = ['red','white','brown','green']))
"""deleting from series and dataframe in pandas"""
# series4 = pd.Series([11,22,33,44,55,66,77,88], index =list('abcdefgh'))
# print(series4)
# print(series4.drop('e')) # this shows that its drops the values but the original series still has them
# print(series4.drop(["e","a"]))
# dt5 = pd.DataFrame(np.arange(25).reshape((5,5)),
#                    index = list('abcde'),
#                    columns = ['red','white','brown','green','black'])
# print(dt5)
# print(dt5.drop(['b','e'])) #to delete a row in a dataframe
# print(dt5.drop('green',axis = 1)) # to drop a column also this is only a copy
# print(dt5.drop('white',axis = 1))
# dt5.drop('red',axis = 1, inplace = True) # to permanently delete a column from the original
# print(dt5)
# dt5.drop(['c'], inplace = True) # permanently delete a row
# print(dt5)
# print()
"""slicing and filtering"""
# print(series1)
# print(series1['c'])
# print(series1[['c','d','e']])
# print(series1['c':'f']) # select labels
# print(series1[4])
# print(series1[2:6])
# print(series1[series1 >40])
# print()
# print(dt5)
# print(dt5['green'])
# print(dt5.loc['a'])  # loc works with dataframes labels while iloc works digits(location of elements)
# print(dt5.loc[['a']]) # look like a dataframe
# print(dt5.loc['b':'e']) # when using labels end of the range is inculded but not in digits
# print(dt5.loc[['b','d'],['green','white']]) # can also use : to get a range of labels
# print(dt5.iloc[2:5, 2:4]) # [row indexing, column indexing]
# print()
# print(dt5[dt5['brown'] > 10])
# print()
# print(dt5.iloc[:,2:5][dt5["black"] >12]) # combining iloc and bool
# print()
"""arithmetic operation to dataframe"""
# dt6 = pd.DataFrame(np.arange(1, 13, 2).reshape((3, 2)), columns=['a', 'b'])
# print(dt6)
# print(10/dt6)
# print(dt6.div(2),dt6.add(3),dt6.sub(1),dt6.mul(3),dt6.pow(2))
# print()
# print(dt6['a'].add(2)) # specific column
# print(dt6.iloc[1].add(2)) # perform for a specific row
# print(dt6['b'] - dt6['a']) # columns operation
# print(dt6.min())
# print(dt6.mean())
# print(dt6)
# print()
# print(dt6.sum())  # this is for row bur for column pass "axis = 'column'" in the sum
# print(dt6.describe())  # returns all the descriptive statistics
# print(dt6.idxmax())  # returns the index of the max
# print(dt6.max() - dt6.min())
# norm_dt6 = (dt6-dt6.mean())/((dt6.max()-dt6.min()))
# print(norm_dt6)# to normalize a data column to a standard scale eg.((x-mean(x))/(max(x)-min(x)))
# norm_series = (series1-series1.mean())/((series1.max()-series1.min()))
# print(norm_series)
"""sorting series and dataframes"""
# series5 = pd.Series(np.random.randint(50, size=10), index=list('jsgjhfsagb'))
# print(series5)
# print(series5.sort_index())  # this a copy for permanent used implace = True
# # this is in ascending but for descending use 'ascending = False'
# print(series5.sort_values())
# dict3["b"] = [3, 5, 8]
# print(dict3)
# dfe = pd.DataFrame(dict3)
# print(dfe)
# print()
# print(dfe.sort_values(by="white"))  # you can sort a particular column by passing this in the sort_values '(by = the
# # column name)'
# print(dfe.sort_values(by=["white", "red"]))  # the first column takes precedence
"""correlation and covariance"""
# correlation is a measure of strength of a relationship between two variables(its between -1 to +1)
# positive means moving in the same direction, while negative means as one increase the other decrease and vice versa
# correlaion becomes weaker when approaching zeroband stronger towards 1 or -1
# des = pd.read_excel('data.xlsx')
# print(des.head())
# dict = {'tv': [230.1, 44.5, 17.2, 151.5, 180.8], 'radio': [37.8, 39.3, 45.9, 41.3, 10.8],
#         'newspaper': [69.2, 45.1, 69.3, 58.5, 58.4], 'sales': [22.1, 10.4, 9.3, 18.5, 12.9]}
# des = pd.DataFrame(dict, index=list('12345'))
# print(des)
# print()
# print(des['tv'].corr(des['sales']))  # checks correlation between the tv and their sales
# print()
# print(des.tv.corr(des.sales))  # or like this
# print()
# print(des.tv.corr(des.radio))
# print()
# print(des.corr())  # correlation matrix
# covariance examines the relationship between two variables by measuring the extent to which the two variables
# change with each other
# print(des['tv'].cov(des['newspaper']))
# print()
# print(des.tv.cov(des.newspaper))
# print()
# print(des.cov())
# the covariance between two variables and itself is the variance of the variable
# variance measure how far a variable is spread out
"""reading data in text format"""
# with open("C:/Users/adedi_tpk1ys1/Downloads/names/yob1880.txt","r") as file1:
#     file_stuff = file1.read()
#     #print(file_stuff)
#     df1 = pd.read_table(file_stuff, sep = ',')
#     print(df1)
# pip3 install --upgrade pandas
# df1 = pd.read_fwf("C:/Users/adedi_tpk1ys1/Downloads/names/yob1883.txt", names = ['Name', 'Gender','DOB'])
# print(df1) # can be used to import text too
# df1 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/names/yob1883.txt", names = ['Name', 'Gender','DOB'])
# print(df1)
# print(df1.head())
# read_table() when your data in the text file is seperated by something else
# eg. df2 = pd.read_table('yob1885.csv',sep = '%') it is seperated by %
# when your data doesn't contain header label you use 'header = None'
# eg.df3 = pd.read_csv('yob1885.csv', header = None)
# "C:\Users\adedi_tpk1ys1\Downloads\names\yob1880.txt"
# or to pass the labels you want
# eg.df3 = pd.read_csv('yob1885.csv',names =['name,'mpg','displacement'])
"""assignment of column to be index,skipping rows in the files,and checking for missing values"""
# df2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/names/yob1883.txt", index_col = 'name')
# print(df2.head())
# df2 = pd.read_table("C:/Users/adedi_tpk1ys1/OneDrive/Documents/woe.txt", sep = ',', index_col = 'Date')
# print(df2)
# print()
# df2 = pd.read_table("C:/Users/adedi_tpk1ys1/OneDrive/Documents/woe.txt", sep = ',', index_col = ['Date','NASDAQ'])
# print(df2)
# print()
"""skipping rows"""
# df2 = pd.read_table("C:/Users/adedi_tpk1ys1/OneDrive/Documents/woe.txt", sep = ',',skiprows = [2,4,10])
# print(df2)
# print()
"""missing data"""
# df2 = pd.read_table("C:/Users/adedi_tpk1ys1/OneDrive/Documents/woe.txt", sep = ',')
# print(df2)
# print(pd.isnull(df2))
# print()
# print(pd.isnull(df2).any())
"""writing data in text format"""
# df3 = pd.read_csv("C:/Users/adedi_tpk1ys1/OneDrive/Documents/woe.txt", index_col = 'Date')
# print(df3)
# df3['average'] = (df3['NASDAQ'] + df3['DOWjones'] + df3['S$P500'])/3
# print(df3.head())
# print()
# df3.to_csv("C:/Users/adedi_tpk1ys1/OneDrive/Documents/wae.txt")
# df4 = pd.read_csv("C:/Users/adedi_tpk1ys1/OneDrive/Documents/wae.txt", index_col = 'Date')
# print(df4)
# print()
# to list file in your directory use listdir() or check the document in the directory

# # print(listdir("C:/Users/adedi_tpk1ys1/Downloads/names"))
# series1 = pd.Series(np.arange(10),index = list('abcdefghij'))
# print(series1)
# series1.to_csv("C:/Users/adedi_tpk1ys1/Downloads/names/series1.csv")
# print(listdir("C:/Users/adedi_tpk1ys1/Downloads/names"))
# print()
"""reading microsoft execl files"""
# df5 = pd.read_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/data.xlsx")
# print(df5.head())
# df6 = pd.DataFrame(df5, columns = (['Date','NASDAQ','DOWjones','S$P500']))
# print(df6.head())
# df6.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/doto.xlsx")
"""Data cleaning and more on missing data"""
# series1 = pd.Series([23,54,np.nan,None])
# print(series1)
# print(series1.isnull())
# print(series1.isna())
# ser2 = pd.Series(['green','black','white',None,'red',np.nan])
# print(ser2)
# print(ser2.isnull())
"""Finding missing data"""
# # there are two ways notnull()/Boolean indexing and dropna()
# print(series1.dropna())
# print(series1[series1.notnull()]) # remember its a  mirror for permanent use implace = True
# df1 = pd.DataFrame([[1,None,3,4,5],[6,None,8,9,10],[11,None,13,14,15]])
# print(df1)
# print(df1.dropna())
# print()
# print(df1[df1.notnull()])
# print(df1.dropna(axis = 1))
# df2 = pd.DataFrame([[6,None,8,8,10],[6,34,88,24,45],[6,None,13,None,15],[6,17,None,19,15,]])
# print(df2)
# print(df2.dropna(how ='all')) # to remove all a row containing all missed data
# # for column we use
# print(df1)
# print(df1.dropna(axis = 1, how ='all'))
# # to remove a rows with less than a particular valid data use
# print(df2)
# print(df2.dropna(thresh = 3)) # sets the minimum number of valid data
# print(df2.dropna(axis = 1,thresh = 3))# for column pass this 'axis = 1' into the dropna
"""filling in missing data"""
# print()
# print(df2.fillna(20))
# print(df2.fillna({0:21,1:32,2:12,3:76,4:10})) # to fill individual column with different values
# print()
# # print(df2.fillna(method = 'ffill')) # to fill with it with the value preceding the missing value
# # print(df2.fillna(method = 'bfill')) # backward filling
# print()
# print(df2.fillna(df2.mean())) # fills with the mean of each column
# print()
"""removing duplicate entries"""
# df4 = pd.DataFrame([[6,74,8,8,10],[7,34,88,24,45],[6,20,13,20,15],[56,17,13,19,15,]])
# print(df4)
# print()
# print(df4.nunique()) # returns the number of distinct values across the row
# print(df4.duplicated())#checks for duplicate
# print(df4.duplicated().any()) # this return only one line to tell if there are duplicates
# print(df4.drop_duplicates()) # to remove the duplicates for rows and
# print()
##get the unique values (rows) by retaining last row
# print(df4.drop_duplicates(keep='last'))
# # get distinct values of the dataframe based on column
# df5 = df4.drop_duplicates(subset = ["Age"]) # pass the column you want
# print(df5)
"""Replacing Values"""
# temp = pd.Series([23,37,999,32,32,28,999,19,24])
# print(temp)
# temp.replace(999,np.nan, inplace =True)
# print(temp)
# temp2 = pd.Series([23,37,999,32,999,28,1000,19,20,-999,24])
# print(temp2)
# print(temp2.replace({999: 29,1000:40,-999: -23})) # to replace multiple unique values with different values
"""renaming columns and index labels"""
# data = pd.DataFrame(np.arange(12).reshape(4, 3), index=['green', 'red', 'black', 'white'],
#                     columns=['one', 'two', 'three'])
# print(data)
# data.rename(index={'green': 'yellow'}, inplace=True)  # renaming row index
# print(data)
# data.rename(columns={'two': 'four'}, inplace=True)  # renaming columns labels
# print(data)
# data.index = data.index.str.upper()  # change the row index to upper format
# print(data)
# data.columns = data.columns.str.title()  # change the columns label to title format
# print(data)
"""filtering outliers"""
# outliers are observation that lies at an abnormal distance from the others
# data2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex2.txt")
# print(data2.head())
# print(data2.describe())
# print(data2[data2.Cards >5])
# data2.loc[4,['Cards']]= 4
# data2.loc[12,['Cards']]= 5
# print(data2)
# print(data2.Cards.any() >5)
"""shuffling and random sampling"""
# ser1 = pd.Series(np.random.randint(20,size=10))
# print(ser1)
# print(ser1.sample(frac=1)) # to shuffle the data using fraction = 1 meaning 100 percent of the data will be returned
# print(ser1.sample(frac=1).reset_index(drop=True)) # to shuffle the values alone, the index will be ordered
# print()
# print(ser1.sample(frac=.8).reset_index(drop=True))# to shuffle but returns only a particular percent of the data.(you can set it)
# # with dataframe random sampling is done through the rows
# print(data2)
# print(data2.sample(frac=.2).reset_index(drop=True))# using percentage
# print(data2.sample(n=5).reset_index(drop=True))# passing a particular number of rows
"""dummy variables"""
# categorical variables need to be converted into dummy so they can be used for statistical modeling and machine learning models
# data2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex4.txt")
# print(data2.head())
# marriage_dummies =pd.get_dummies(data2['marital'])
# gender_dummies = pd.get_dummies(data2['sex'])
# print(gender_dummies)
# data_with_dummies =data2.join([gender_dummies,marriage_dummies]) # the dummy with the original to another variable
# print(data_with_dummies)
"""string object methods"""
# text1 = 'jone,sam,jake'
# text2 = 'sam will go to the school today'
# print(text1.split(','))
# print()
# words = [x.strip()for x in text1.split(',')] # when there are spaces inbetween the strings
# print(words)
# print()
# print(text2.split(' '))
# text3 = ['sam','yahoo.com']
# print('@'.join(text3)) # to join strings
# print('school' in text2) # using in to find
# print(text2.index('school'))# for find the index of a string or
# print(text2.find('today'))
# print(text2.find('jone')) # it returns '-1' when the string is not found
# print(text2.count("a")) # to check the occurence of a particuler substring
# print(text2.count('to'))
# # replace(,) to replace a particular substring
# print(text2.replace('a','e'))
"""hierarchical indexing"""
# data wrangling is a wide range of methods which are used to prepare the data for further analysis
# various type of wrangling are combining,joining and re-arranging
# Hierarchical indexing means to have multiple index levels, whether it is a row index or a column index
# Hierarchical indexing is used for regrouping data
# ser1 = pd.Series(np.random.randn(9),index = [['2010','2010','2010','2011','2011','2011','2012','2012','2012'],
#                                              ['one','two','three','one','two','three','one','two','three']])
# print(ser1)
# print(ser1.index) # to show index
# print()
# print(ser1['2010'])
# print(ser1['2011':'2012'])
# print(ser1.loc[['2010','2012']])
# print(ser1.loc[:,'two'])# all the values in postion 'two' for the three years
# print()
# data1 = ser1.unstack() # helps to convert a multi_indexed series into a dataframe
# print(data1)
# print(data1.stack())# converts a dataframe to a multi-indexed series
# data2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex5.txt")
# print(data2.head())
# multi_data = data2.set_index(['year','quarters']) # to set a particular column as a second index
# print(multi_data)
# print(multi_data.index) # check index
# print(multi_data.loc[2010])# to get the subset of all the values of 2010
# print()
# print(multi_data.loc[(2010,'A')]) # to get the subset of the values of 2010 and in 'A' alone
# data3 = pd.DataFrame(np.random.randint(100,size=(4,4)),# to create multi-labels foe the column
#                      columns = [['green','green','black','black'],['one','two','one','two']])
# print(data3)
"""reording and sorting index levels"""
# data4 = pd.DataFrame(np.random.randint(100,size=(6,4)),# to create multi-labels foe the column
#                      index = [['green','green','black','black','yellow','yellow'],['one','two','one','two','one','two']])
# print(data4)
# data5 = data4.swaplevel(0,1) # to swap the firdt column of the index with the second one
# print(data5)
# print(data5.sort_index(level = 0)) # to help sort the new first to fit(first index is 0 while second is 1)
# print(data4.swaplevel(0,1).sort_index(level = 0)) #easier way do them together
"""summary statistics by level"""
# # applying descriptive statistics by level to multi-indexed dataframe
# print(data4)
# print(data4.mean())
# print(data4.index.names) # to check the index name
# data4.index.names = ['color','number'] # assigning names to the indexes
# print(data4)
# print(data4.mean(level = 'color'))
# print(data4.groupby(level=1).median()) # new version of python only allow this format(0 for first and 1 for second)
# print(data4.groupby(level = 0).describe()) # for colors
"""indexing with columns in dataframe"""
# data2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex5.txt")
# print(data2.head())
# print(data2.columns)
# print()
# data4 = data2.set_index('year')
# print(data4)
# print()
# data5 = data2.set_index(['year','quarters']) # making a multi-index dataframe
# print(data5)
# print(data2.set_index('year', drop = False)) # to leave the column even though you set it as an index
# # reset_index() to reset the index to it original position
# print(data5.reset_index())
"""merging datasets based on common columns"""
# merging two dataframe using a common columns
# data1 = pd.DataFrame({'states':['california','georgia','florida','arizona'],'population':[40,10,21,7]})
# print(data1)
# print()
# data2 = pd.DataFrame({'states':['arizona','colorado','indiana','florida'],'area':[113,1044,36,65]})
# print(data2)
# print()
# data3 = pd.merge(data1,data2) # merge only common values in the common column. this is called intersection merging
# print(data3)
# print()
# print(pd.merge(data1, data2, on = 'states')) # for choosing a particular column for the merging
# print()
# print(pd.merge(data1,data2, how = 'outer')) # this includes every of the values
# print()
# print(pd.merge(data1,data2, how = 'left')) # here you can specific from which data it should include all e.g.here 'left'
# print()
# data3 = pd.DataFrame({'states_name':['california','georgia','florida','arizona'], 'population':[40,10,21,7]})
# print(data3)
# print()
# print(pd.merge(data3, data2, left_on = 'states_name',right_on = 'states')) # here when you want to merge but with different names
# print()
# data4 = pd.DataFrame({'states':['california','georgia','florida','arizona'],'population':[40,10,21,7],'water':[4,6,7,2]})
# print(data4)
# print()
# data5 = pd.DataFrame({'states':['arizona','colorado','indiana','florida'],'area':[113,104,36,65],'water':[2,8,3,7]})
# print(data5)
# print(pd.merge(data4,data5, on = 'states', how = 'outer'))
# print(pd.merge(data4,data5, on = 'states', how = 'outer', suffixes = ('_data4','_data5'))) # to make it easy to
# understand we label the two with the suffix (when there are common columns)
"""merging datasets on  common index"""
# data1 = pd.DataFrame(np.random.randint(100, size =(4,3)),index = ['b','c','e','f'],
#                      columns = ['green','red','white'])
# print(data1)
# data2 = pd.DataFrame(np.random.randint(100, size = (3,4)),index =['a','b','c'],
#                      columns = ['blue','yellow','purple','black'])
# print(data2)
# print(pd.merge(data1,data2,left_index = True, right_index = True))  # for the common rows
# print(pd.merge(data1,data2,left_index = True, right_index = True, how = 'outer')) # for all
"""concatenating series and dataframe along an axis"""
# ser1 = pd.Series([1,2,3],index =['a','b','c'])
# ser2 = pd.Series([4,5,6],index= ['d','e','f'])
# print(pd.concat([ser1,ser2]))
# print(pd.concat([ser1,ser2], axis = 1)) # concatenating along the columns
# ser3 = pd.Series([7,8,9], index = ['a','b','g'])
# print(pd.concat([ser1,ser3], axis = 1))
# print(pd.concat([ser1,ser3], axis = 1, join = 'inner')) # to eliminate non available values
# """dataframe"""
# dat1 = pd.DataFrame({'states':['california','georgia','florida','arizona'],'population':[40,10,21,7]})
# dat2 = pd.DataFrame({'states':['hawaii','colorado','indiana','alaska'],'population':[1.5,10.4,5.7,0.7]})
# print(pd.concat([dat1,dat2], ignore_index = True)) # you pass the ignore_index so the result is ordered
# dat3 = pd.DataFrame({'area':[113,104,36,65],'water':[23,54,12,45]})
# print(pd.concat([dat1,dat3], axis = 1))# just arranging the columns next to each other cause there are no common columns
"""reshaping/rearranging by stacking and unstacking"""
# three common ways: stacking, melting, pivoting
# stacking is moving the innermost column index to become the innermost row index
# unstacking is moving the innermost row index to become the innermost column index
# data =pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex6.txt",
#                   index_col = 'branch number')
# print(data.head())
# data2 = data.stack() # to change the columns to rows not the index
# print(data2)
# print(data2.unstack()) # using unstack to return it to its previous format
# print(data2.unstack(0)) # pandas recogonizes the outer index as 0 and the inner index as 1
# print(data2.unstack('branch number'))
# print()
# frame1 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex5.txt",
#                   index_col = ['year','quarters'])
# print(frame1)
# print(frame1.stack())
# print(frame1.unstack())
# print(frame1.unstack(0))# it unstack using the index passed as column with the original inner columns
"""reshaping by melting"""
# it used to convert multiple columns into a single column
# frame1 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex7.txt")
# print(frame1.head())
# data = frame1.melt(id_vars = 'year') # the column passed is the column you do not want to melt
# print(data)
# data1 = frame1.melt(id_vars = 'year', var_name ='sales', value_name ='amount') # this is to name the generated columns
# print(data1)
# frame2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex8.txt",
#                      index_col = 'date')
# print(frame2.head())
# data1 = frame2.melt(var_name ='company', value_name ='closing price')
# print(data1)
"""personal exercise"""
# df2 = pd.read_table("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/woe.txt", sep = ',', index_col = 'Date')
# print(df2.head())
# # to retain the index from the dataframe pass 'ignore_index = False'
# data1 = df2.melt(var_name ='company', value_name ='closing price',ignore_index = False)
# print(data1.head())
# # to sort the index
# print(data1.sort_index())
"""reshaping by pivot"""
# turning one column into multiple columns in dataframe(long to wide)
# frame2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex9.txt")
# print(frame2)
# data2 = frame2.pivot(index ='year', columns = 'sales', values = 'amount')
# print(data2)
# frame2 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex10.txt")
# print(frame2)
# data2 = frame2.pivot(index= 'date',columns='company',values= 'closing price')
# print(data2)
# # or you can shorten it
# data3 = frame2.pivot('date','company','closing price')
# print(data3)
# # setting multi-indexed dataframe with the inner index is the column that will be pivoted
# data4 = frame2.set_index(['date','company'])
# print(data4)
# print( data4.unstack('company'))
"""Data visualization:matplotlib"""
# uses of data plotting are: data exploration and advanced data modeling
# common libraries for data visualization are: matplotlib, pandas,seaborn
# import matplotlib as mpl
import matplotlib.pyplot as plt

# array = np.arange(10, 20)
# print(array)
# print(plt.plot(array))
# print(plt.show()) # to show the graph using pycharm
"""creating figures and subplots"""
# subplot are used  to display multiple plots in one figure.(for comparing them)
# fig = plt.figure()  # this creates a random figure
# print(fig)
# fig = plt.figure(figsize = (8,6)) # this is used to set the exact size of the figure
# print(fig)
# print()
# ax1 = fig.add_subplot(2,2,1) # the 2,2 means  the gird of the subplot contains
# # two rows and columns and 1 means its the first subplot
# print(ax1)
# print(plt.show())
# fig2 = plt.figure(figsize=(6, 4))
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)
# ax1.plot([5, 6, 4, 3, 7, 8])  # to plot in the graphs
# ax2.plot(np.random.randn(20).cumsum())  # to plot the cumulative sum of a set of random numbers
# ax3.plot([23, 4, 77, 89, 35])
# ax4.plot([-23, -6, 4, -76, -40])
# print(fig2)
# plt.show()
# an easier way to create subplots
# fig, ax = plt.subplots(3,3) # 3 rows and 3 columns
# plt.show()
# fig,ax = plt.subplots(3,3, figsize = (8,6))
# ax[0,0].plot([1,3,5,67,3]) # to access the individual plot in the multi(rows are 0,1,2,3... and columns the same)
# ax[1,2].plot([345,6,76,9,3,6,8])
# fig.subplots_adjust(hspace=0.4,wspace=0.4) # to create space between the subplots
# plt.show()
"""changing colors, markers and linestyle"""
# y = np.random.randn(20).cumsum()
# print(y)
# plt.plot(y)
# x = np.arange(0,200,10)
# plt.plot(x,y)
# plt.plot(x,y,color='b') # to change colors(ploting it in a x and y graph)
# # b for blue, r for red, m for magenta, y for yellow, k for black,w for white, g for green
# plt.plot(x,y,color= 'g',marker= 'd')# this is used to set marker along the line
# # different markers are o for circle, s for square, p for pentagon, d for diamond, v for triangle,
# plt.plot(x,y,color= 'g',marker= 'o',linestyle= '--')
# # '--' for dash line, ':' for dotted line
# #combining all in a simpler form
# plt.plot(x,y,'ks:')# k for color black, s for square marker and colon for dotted lines
# plt.show()
"""customizing ticks and labels"""
# data = np.random.randn(200).cumsum()
# fig, ax = plt.subplots(1, 1)
# ax.plot(data)
# ticks = ax.set_xticks(range(0, 201, 10))  # the x before ticks is for x-axis(y for y-axis) (remember range format)
# ticks = ax.set_xticks([0,50,100,150,200]) # set the ticks by listing them out
# # (you have to set the ticks first and the len of ticks should match the length of labels)
# labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'])  # to use a label for the axis
# # for in case of overlapping labels use rotation to pass the degree to rotate the labels
# labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],rotation = 60)
# ax.set_xlabel('Cumulative Sum') # to label the axis of the graph (same for y-axis change to 'y')
# ax.set_ylabel('Values') # to label the y-axis
# ax.set_title('Cumulative Sum for Random Numbers') # to label the graph
# plt.show()
"""adding legends"""
# #we use legends when we plot multiple data on the same plot or multiple subplots on the same figure
# #(they are used to identifying plot components. i.e. which graph belong to which dataset)
# data1 = np.random.randn(200).cumsum()
# data2 = np.random.randn(200).cumsum()
# data3 = np.random.randn(200).cumsum()
# fig,ax = plt.subplots(1,1)
# ax.plot(data1, label= 'Data1')
# ax.plot(data2, label= 'Data2')
# ax.plot(data3, label= 'Data3')
# ax.set_xlabel('Cumulative Sum')
# ax.set_ylabel('Values')
# ax.set_title('Cumulative Sum for 3 Random set of Numbers')
# ax.legend() # to show the legend after passing the label for each above
# ax.legend(loc ='best') # to pass the position of the legend
# # best for best suitable position, lower right, upper right, lower left, upper left
# plt.show()
"""adding texts and arrows on a plot"""
# x = np.arange(0,10,0.1)
# y = np.sin(x)
# fig,ax = plt.subplots(1,1)
# ax.plot(x,y)
# ax.text(4,0.7,'y = sin(x)',fontsize = 15)# (4,0.7) is the corodinate in which you want the text to be
# # if you dont want to set the coordinate you can use defalut point in relation to the square
# #like (0.5,0.5) is the center of the plot,(0,1) is the upper left corner,(1,0) is lower right corner,(1,1) is upper right, etc passing it with transform=ax.transAxes
# ax.text(1,1,'y = sin(x)',fontsize = 15, transform = ax.transAxes)
# ax.arrow(0.5,1,0.6,0,width=0.05) # the (0.5,1) is the coordinate for the arrow
# # while the last deal with the length and direction of the arrow  and the width how wide the arrow is
# ax.arrow(6.8,1,0.6,0,width=0.05)
# ax.arrow(3.5,-0.95,0.6,0,width=0.05)
# ax.text(0.5,1.02,'rise',fontsize = 15)
# ax.text(6.8,1.02,'rise',fontsize = 15)
# ax.text(1.0,-0.95,'depression',fontsize = 15)
# plt.show()
"""adding annotations and drawings"""
# # to draw attention to special features or important points on the graph
# x = np.arange(6)
# y = np.array([23, 34, 65, 78, 51, 55])
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, y, 'ko--')
# ax.annotate('maximum', xy=(x[3], y[3]), xytext=(3, 50),
#             arrowprops=dict(facecolor='green', shrink=0.1))
# # first you pass the text, then the location of the arrow(you can also pass the actual values of the location like(3,
# # 78)), then the location of the text(this controls the length of the arrow too) then pass arrowprop to chose color
# # and the other properties of the arrow
# ax.annotate('Minimum', xy=(x[0], y[0]), xytext=(0, 50),
#             arrowprops=dict(facecolor='g', shrink=0.1))
# plt.show()
"""adding shapes to a plot"""
# fig, ax = plt.subplots(1, 1)
# circle = plt.Circle((0.5, 0.5), 0.2,
#                     fc='m')  # (this create the circle) first x and y coordinate, then radius and then the color of circle
# ax.add_patch(circle)  # adds the circle tp the plot
# rect= plt.Rectangle((0.3,0.5),0.5,0.7,fc='y',alpha =0.5)# first is the lower left coordinates,lenght and width, color, alpha is to adjust the transparency
# ax.add_patch(rect)
# tri = plt.Polygon([[0.25,0.25],[0.55,0.5],[0.4,0.7],[0.2,0.6]],fc= 'r',alpha=0.8)
# # use polygon that draws base on the number of vertices passed(can be used for any other polygon)
# ax.add_patch(tri)
# plt.show()
"""saving plots to file"""
# x = np.arange(6)
# y = np.array([23, 34, 65, 78, 51, 55])
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, y, 'ko--')
# ax.set_title("My first Figure")
# ax.set_xlabel('x axis')
# ax.set_ylabel('y axis')
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/firstfig.svg") # Scalable Vector Graphic(keep image quality as it scale up or down)
# # or plt.savefig("C:/Users/adedi_tpk1ys1/Documents/firstfig.png")Portable Graphics( the quality is lower when you zoom in)
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/firstfig2.png", dpi = 400) # to improve the png use
# #or jpeg: plt.savefig("C:/Users/adedi_tpk1ys1/Documents/firstfig.jpeg")
# #or pdf: plt.savefig("C:/Users/adedi_tpk1ys1/Documents/firstfig.pdf")
# from os import listdir
# print(listdir("C:/Users/adedi_tpk1ys1/Documents/"))
# plt.show()
"""line plot with dataframe"""
# advantages of pandas for ploting are: flexibility when dealing with multiple columns,plotting options
# are automatically handled by pandas, easier plotting and shorter codes and more esthetic values for plotting
"""pandas library"""
# ser1 = pd.Series(np.random.randn(10).cumsum())
# ser1.plot()
# data1 = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex7.txt", index_col = 'year')
# print(data1)
# data1.plot()
# data1.plot(subplots = True) # to return individual plots
# data1.plot(subplots = True,layout= (1,3)) # to return horizontal individual plots
# data1.plot(subplots= True, layout= (1,3), figsize= (12,6)) # to increase the size
# data1.plot(subplots= True, layout= (1,3), figsize= (12,6), legend = False) # to remove all legends
# data1.plot(subplots= True, layout= (1,3), figsize= (12,6), title= 'Company Sales for Three items') # to add title
# data1.plot(subplots= True, layout= (1,3), figsize= (12,4,),
#            title= 'Company Sales for Three items', sharey= True) # to make them share the y-axis
# data1.sale1.plot() # to plot only one column
# plt.show()
"""bar plots with dataframes"""
# # two: vertical we use 'plot.bar()' and horizontal we use 'plot.barh()'
# data = pd.Series([34, 76, 12, 89, 45], index=list('ABCDE'))
# data2 = pd.Series([34, 76, 12, 89, 45], index=list('ABCDE'))
# print(data)
# data.plot.bar()
# # data2.plot.barh()
# # you can customize using 'rot' to rotate, 'color','alpha' for transparency
data1 = pd.read_csv(
    "C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex6.txt",
    index_col='branch number')
# data1.plot.bar(title='Sales from different Branches', rot=0.3)
# data1.plot.bar(title='Sales from different Branches', stacked=True, rot=60)  # to create stacked plot
data1.plot(kind= 'barh') # another way to call bar plots
plt.show()
"""bar plots and seaborn"""
import seaborn as sns

# frame = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex5.txt")
# print(frame)
# barplot(x= array, y= array) or as a dataframe (barplot(data= dataframe))
# sns.barplot(data= frame)
# sns.barplot(data= frame[['sale1','sale2','sale3']]) # confidence interval
# sns.barplot(data= frame[['sale1','sale2','sale3']], ci = False) # to remove confidence interval
# sns.barplot(x= frame.year, y= frame.sale1,ci= False) # to pick specific columns as x and y-axis(uses average of each column)
# sns.barplot(x= 'year', y= 'sale1',data= frame,ci= False) # to pass individually(it only takes the data and plot)
# sns.barplot(x= frame['year'], y= frame['sale1'],ci= False) # another way to return individual columns
# sns.barplot(x= frame['sale1'], y= frame['year'],orient='h',ci= False) # use "orient='h'" to turn it into an horizontal plot
# sns.barplot(x= frame['year'], y= frame['sale1'],hue = frame['quarters'])# ,title='Quarterly sales for three Years'
# # when dealing with real values confidence interval is not shown
# plt.show()
"""histograms and density plots"""
# #histogram represent the distribution of the data as it displays the shape and the spread of a continuous data
# #changing bins changes the shape. bins is usually between 5 and 20 depending on the size and range of the data
# #bins is the number of bar you have
# frame= pd.read_csv("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex13.txt")
# print(frame.head())
# frame['age'].plot.hist(bins = 12)
# #can pass color, edgecolor, linewidth
# frame['age'].plot.hist(bins = 12,fc= 'r',edgecolor='k',linewidth=1)
# frame['wage'].plot.hist(bins = 15,fc= 'm',edgecolor='k',linewidth=1)
# """density plot"""
# #used to visualize the distribution of the data(advantage is that it is not affected by number of bins)
# frame['wage'].plot.density()
# #another way to call histogram and density plot
# sns.displot(frame['wage'],bins=12,color='b')# another way for histogram
# sns.displot(frame['wage'],kind= 'kde',color='b') # another way for density plot
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/first.svg")
# plt.show()
"""scatter plots and pair plots"""
# #they are very common in data science and statistical modeling,
# #scatter plot is also useful to examine the relationship between two data series
# frame = pd.read_csv(
#     "C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/boston.txt")
# print(frame.head())
# sns.regplot(x='medv', y='crim', data=frame)
# #negative relationship is shown when an increase one result in a decrease of the other and vice versa for positive
# #it adds a regression line to the plot which represent the data prediction model
# #the blue line around the regression line is confidence interval which measures the accuracy of the prediction
# sns.regplot(x='medv', y='crim', data=frame, ci= False)# to remove confidence interval
# sns.regplot(x='medv', y='crim', data=frame, ci=False, fit_reg=False)  # to remove regression line
# #pairplot is used to generate the scatter plot matrix
# sns.pairplot(frame)
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/scatter matrix.svg")
# sns.pairplot(frame,diag_kind='kde')# to change the histogram in the matrix to density plot
# plt.show()
"""factor plots for categorical data"""
# frame = pd.read_csv("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/credit.txt")
# print(frame.head())
# sns.catplot(x='Gender',y='Income',data= frame)
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/catplot.svg")
# sns.catplot(x='Gender',y='Income',data= frame, kind= 'bar',ci=False)# for getting it in bar plot
# sns.catplot(x='Gender',y='Income',data= frame, kind= 'box') # to get boxplot
# sns.catplot(x='Ethnicity',y='Income',data= frame, kind= 'bar',ci=False,hue='Gender')
# plt.show()
# """Time Series Data"""
# this is any dataset in which data is observed or measured at many points in time
from datetime import datetime
from datetime import timedelta

# time_now = datetime.now()
# print(time_now)
# print(time_now.year)
# print(time_now.month)
# print(time_now.day)
# print(time_now.hour)
# time1= datetime(2022,9,15)
# time2= datetime(2022,10,14)
# time_dif= time2 -time1
# print(time_dif)
# print(time_dif.days)
# print(time1 + timedelta(23))
# print(time1 - timedelta(20))
"""converting between string and Datetime"""
# time_str = '2022,09,12'
# time1= datetime.strptime(time_str,'%Y,%m,%d')# you pass the format of the date you want to change inculding the seperator
# print(time1)
# time_str2= '2022-08-13'
# time2= datetime.strptime(time_str2,'%Y-%m-%d')
# print(time2)
# df = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex8.txt")
# print(df.head())
# print(df.date)
# df['date']= pd.to_datetime(df['date']) # to convert the date to timeseries
# print(df.date)
# print(df.head())
"""Data time series"""
# df = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex15.txt")
# print(df.head())
# df['date']= pd.to_datetime(df['date'])
# print(df.head())
# print(df.dtypes )
# df= df.set_index('date') # setting the index as date outside the read function
# print(df.head())
# #combining it into one line(parse_dates converts the date column into a time format )
# df2= pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex15.txt",
#                  parse_dates=['date'],index_col='date')
# print(df2.head())
# print(df2.loc['2021']) # slicing the date column
# print(df2.loc['2021-4'])
# print(df2.loc['2019':'2021'])
# print(df2.loc['2019-06':'2020'])
# print(df2.index.is_unique) # to check for duplicate data
"""Generating date ranges"""
# df2 = pd.read_csv(
#     "C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex16.txt")
# print(df2.head())
# index = pd.date_range(start='2020-05-01', end='2020-05-15')  # creating a data range of dates
# print(index)
# df2 = df2.set_index(index)
# print(df2.head())
# index2 = index = pd.date_range(start='2020-05-01', periods=15)
# print(index2)
# print(len(df2.google))  # to get the length of rows in a dataframe (len(df)) and also foe each column
# index3 = pd.date_range('2020-01-01', '2020-12-15', freq='BM')  # to return the late days of each month passing 'BM'
# print(index3)
# index4 = pd.date_range('2020-01-01', '2020-12-15', freq='B')  # to return business days no weekends
# print(index4)
# index5 = pd.date_range('2020-01-01', '2020-12-15', freq='8h')  # to generate hourly periods
# print(index5)
# index6 = pd.date_range('2020-01-01', '2020-12-15', freq='5h30min')  # hour and minutes
# print(index6)
# index7 = pd.date_range('2020-01-01', '2020-12-15', freq='WOM-2THU')  # to get third friday of the month(can do other
# # variations) WOM stands for Week Of Month
# print(index7)
# # D for day, B for business-day, H for hours, T or min for minutes, S for second, M for month end, BM for business
# # month end, MS for month-begin, BMS fo business month-begin,WOM for week of the month,
"""shifting data through time"""
# # data shifting means moving data backwards and forward through timr while leaving the data index unchanged
# # moving backward is lagging while moving forward is leading . shifting introduces missing values
# df2 = pd.read_csv(
#     "C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex17.txt",
#     parse_dates=True, index_col='date')
# print(df2)
# print(df2.shift(periods=2))  # lag price is shifting it backwards (it reoves from the back)
# print(df2.shift(2))  # or like this
# print(df2.shift(-2))  # forward shifting
# df2['lag1'] = df2['apple'].shift(1)  # assigning a new column and adding the lag series to it
# df2['lag2'] = df2['apple'].shift(-1)  # adding another column of forward shifting
# df2.dropna(inplace=True) # droping the missing values
# # percentage change in price this is percentage change in the value of a stock over a single day trading
# # it is calculated by the price of the current day over the price of the previous day all together minus 1
# df2['percent_change']= (df2['apple']/df2['apple'].shift(1))-1
# df2.dropna(inplace=True)
# print(df2)
"""handling time zone"""
import pytz

# print(pytz.common_timezones)
# data = pd.date_range('2022-07-01 13:30', periods=10, freq='h')
# print(data)
# print(data.tz)  # to check the timezone
# data1 = pd.date_range('2022-07-01 13:30', periods=10, freq='h', tz='UTC') # to create a timeseries with a time zone
# print(data1)
# df = pd.read_csv(
#     "C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/ex18.txt",
#     parse_dates=True, index_col='time')
# print(df)
# print(df.index.tz)  # to check the timezone of a dataframe
# df_utc = df.tz_localize('UTC')  # to assign a timezone to a dataframe
# print(df_utc)
# print(df_utc.tz_convert('Africa/Lagos'))  # converting to a different timezone
"""resampling and frequency conversion"""
# # resampling means converting the index from one date frequency to another one
# # example is aggregating lower frequency data(daily data) to higher frequency data(monthly data)
# df = pd.read_csv(
#     "C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/apple.txt",
#     parse_dates=True, index_col='Date')
# print(df.head())
# df2= df.resample('W').mean()
# print(df2)  # calculating the average price per week using mean
# print(df.resample('M').mean())  # W for weekly, Y for yearly and M  for monthly
# print(df.resample('M', kind='period').mean())  # to remove the dates and return only years and month
# df2.to_csv('C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/time series.txt')
# sales= pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/sales.txt",
#     parse_dates=True, index_col='Date')
# print(sales)
# print(sales.resample('M', kind='period').sum())  # using sum to calculate monthly total
"""rolling and moving windows"""
# # rolling is like having a window sliding over your time series data and extracting out all data seen through the window
# # it is often used in timeseries to: check for its stability over time and explore the main trend of the data
# apple= pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & Seaborn/apple.txt",
#     parse_dates=True, index_col='Date')
# print(apple.head())
# apple['Close'].plot()
# print(apple.rolling(window=3).mean()) # 3 here is the number of data in the moving window
# # ,it needs an aggregate function(mean) and the average of the first to rows can't be calculated because
# # the window was set to 3(its like telling it to calculate the average of the number of windows passed)
# apple['Close'].rolling(window=10).mean().plot() # to get moving average
# plt.show()
# air= pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & "
#                  "Seaborn/airpassenger.txt",parse_dates=True, index_col='Month')
# print(air.head())
# air['Passengers'].plot(figsize=(8,5))
# air['Passengers'].rolling(window=10).mean().plot()
# plt.show()
"""Real World Data Analysis"""
"""housing dataset """
# mian objective is predicting the future sales of houses
# lotArea- lot size in square feet, OverallQuall- rating the overall material and finishing of the house,
# YearBuilt- original construction date, TotalBsmtSF- total square feet of basement area,
# GrLivArea-above ground living area square feet, FullBath- full bathroom above grade, BedroomAbvGr-bedrooms above grade
# KitchenQual- kitchen quality, TotRmsAbvGrd- total rooms above grade, GarageArea- size of garage in square feet
# Data exploration are in five main steps
# 1. Understanding the variables
# 2. Explore the dependent variable(the targe variable we are trying to predict)
# 3. Investigate the relationship between the dependent and the independent variables
# 4. Data cleaning
# 5. Checking statistical assumptions(e.g. check for the data normality)
from scipy import stats  # this is a library for scientific analysis with python

# df = pd.read_csv("C:/Users/adedi_tpk1ys1/Downloads/Python Bootcamp for Data Science 2021 Numpy Pandas & "
#                  "Seaborn/housing.txt")
# print(df.head())
"""step1"""
# print(df.shape)  # return the number of rows and columns
# print(df.info())  # if the value of non-null is equal to total row then no missing values
# print(df.isnull().sum())  # return the sum of missing values in each of the columns
"""step2"""
# print(df['SalePrice'].describe())  # check the descriptive analysis of the target variable
"""step3"""
# print(df.corr())  # check the correlation of the target variable and the independent variables(general effect)
# #  select the variables that have a greater effect(here: OverallQual,TotalBsmtSF, GrLivArea, GarageArea)
# # virtually examine the relation of OverallQual,TotalBsmtSF, GrLivArea,GarageArea using pairplot()
# columns= ['OverallQual','TotalBsmtSF','GrLivArea','GarageArea','SalePrice']
# sns.pairplot(df[columns])
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/housing_pairplot.png")
"""step4"""
# # using plot of each variable to check for outliers or abnormal variables
# df.plot.scatter('OverallQual','SalePrice') # shows a postive relationship
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/scatterplot1.png")
# df.plot.scatter('TotalBsmtSF','SalePrice') # here we have some outliers
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/scatterplot2.png")
# # we have to delete the outliers in this case. so we use boolean index to see the outliers
# print(df.loc[(df['TotalBsmtSF']>3000)&(df['SalePrice']<300000)])# we check for the index values
# df= df.drop([332,523,1298])# we use the index values here to delete them
# df.plot.scatter('TotalBsmtSF','SalePrice')
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/clean_scatterplot2.png")
# print(df['TotalBsmtSF'].corr(df['SalePrice'])) # we check the correlation again and see that its higher without the outliers
# df.plot.scatter('GrLivArea','SalePrice')
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/scatterplot3.png") # we see outliers but they correlation
# # with the normal trend so we leave them
# df.plot.scatter('GarageArea','SalePrice')
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/scatterplot4.png")
# print(df.loc[(df['GarageArea']>1200)&(df['SalePrice']<300000)])# we check for the index values of the outliers
# df= df.drop([581,1061,1190])
# df.plot.scatter('GarageArea','SalePrice')
# plt.savefig("C:/Users/adedi_tpk1ys1/Documents/clean_scatterplot4.png")
"""step5"""
# #to check normality assumption we check the histogram or density plot
# sns.histplot(df['SalePrice'], kde=True)  # we can see it's skew to the right(this is considered a violation of the
# #assumption). The is considered subjective so we use other method like probability plot
# #for a probability plot this to be considered normally distributed the blue dots are to be on the red lines
# stats.probplot(df['SalePrice'],plot=plt)  # we see it is not normally distributed
# # it depends on which statistical analysis method so require it some don't those like linear regression models,
# # logistic regression models, linear discriminant analysis require the data to be normally distributed how to solve
# # this problem of data normality violation is by data transformation using log-transformation or
# # square root-transformation
# df['SalePrice']= np.log(df['SalePrice'])
# # sns.histplot(df['SalePrice'], kde=True)
# stats.probplot(df['SalePrice'],plot=plt)
# plt.show()
"""extra"""
# sns.histplot(df['GrLivArea'], kde=True)
# plt.show() # repeat the previous process
"""categorical data"""
# print(df['KitchenQual'].value_counts()) # to see the number of  values in the column
# kitchen= pd.get_dummies(df['KitchenQual'])
# print(kitchen.head())
# df= df.join(kitchen) # to add the dummy variables to the original dataframe
# print(df.head())
# df= df.drop(['KitchenQual'], axis=1) # axis refers to a column
# print(df.head())
