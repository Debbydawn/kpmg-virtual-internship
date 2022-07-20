import xlrd
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt



data = pd.read_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_VI_New_raw_data_update_final.xlsx",
                     sheet_name='Transactions', index_col=0)
data = data.iloc[:, 0:13]  # selecting the columns with data

data1 = pd.read_excel(
    "C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_VI_New_raw_data_update_final.xlsx",
    sheet_name='NewCustomerList')
data1 = data1.iloc[:, 0:16]  # selecting the columns with data

data2 = pd.read_excel(
    "C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_VI_New_raw_data_update_final.xlsx",
    sheet_name='CustomerDemographic', index_col=0)
data3 = pd.read_excel(
    "C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_VI_New_raw_data_update_final.xlsx",
    sheet_name='CustomerAddress', index_col=0)

"""Using pandas library to determine the excel data datatype by iterating the rows"""
for dtype in data.dtypes.iteritems():
    print(dtype)
print()

""" changing the date to a regular date format"""
data['product_first_sold_date'] = pd.to_datetime(data['product_first_sold_date'], unit= 's')
print(data['product_first_sold_date'].head())

data['transaction_date'] = pd.to_datetime(data['transaction_date'], unit= 's')
print(data['transaction_date'].head())

print(data.info())
print(data1.info())
print(data2.info())
print(data3.info())

# """checking for missing data"""
# def missing(df):
#     for col in df.columns:
#         miss = df[col].isnull().sum()
#         if miss > 0:
#             print("{} has {} missing value(s)".format(col,miss))
#         else:
#             print("{} has No missing value!".format(col))
# print(missing(data))
# print(missing(data1))
# print(missing(data2))
# print(missing(data3))
# """Check for duplicates"""
# def duplicate_columns(df):
#     for col in df.columns:
#         dup = df.duplicated().any()
#         print("{} has {} duplicated value(s)".format(col,dup))
# print(duplicate_columns(data))
# print(duplicate_columns(data1))
# print(duplicate_columns(data2))
# print(duplicate_columns(data3))
# # """Return completeness percentage"""
# def completeness_percent(df):
#     completion = df.isna().sum().sum() / df.size * 100
#     return "The data has a completion percentage of", completion
# print(completeness_percent(data))
# print(completeness_percent(data1))
# print(completeness_percent(data2))
# print(completeness_percent(data3))
# """check for the uniqueness of each columns"""
# def unique_col(df):
#     for col in df.columns:
#         uni= df.nunique()
#     return uni
#
# print(unique_col(data))
# print(unique_col(data1))
# print(unique_col(data2))
# print(unique_col(data3))
# print()
# # def value(df):
# #     for col in df.columns:
# #         count= df[col].value_counts()
# #     return count
# # print(value(data))
#
# print(data['order_status'].value_counts())
# print(data['brand'].value_counts())
# print(data['product_first_sold_date'].value_counts())
# print(data['product_line'].value_counts())
# print(data['product_class'].value_counts())
# print(data['product_size'].value_counts())

"""CLEANING OF DATA FOR ANALYSIS"""
# # print('Cleaning the Data and analysis')
# # print()
# # print("Cleaning Customer Demographic Data")
# data2 = data2.drop(['first_name', 'last_name', 'default', 'job_title'], axis=1)
# data2['gender'].replace(['F', 'Femal'], 'Female', inplace=True)
# data2['gender'].replace('M', 'Male', inplace=True)
# data2['gender'].replace('U', 'Unknown', inplace=True)
# data2['deceased_indicator'].replace(['N'], 0, inplace=True)
# data2['deceased_indicator'].replace(['Y'], 1, inplace=True)
# data2['owns_car'].replace('Yes', 1, inplace=True)
# data2['owns_car'].replace('No', 0, inplace=True)
# data2 = data2[data2['DOB'] != data2.DOB.min()]
# data2[data2['deceased_indicator'] == 0]
# data2 = data2.drop(['deceased_indicator'], axis=1)
# data2_clean = data2.dropna()
# # print(data2_clean.head())
# # print("Generating the Age and Age class columns")
# data2_clean['age'] = (dt.datetime.now() - data2_clean['DOB']) / np.timedelta64(1, 'Y')
# data2_clean['age_class'] = ((round(data2_clean['age'] / 10)) * 10).astype(int)
# # print(data2_clean.head())
# # print("Cleaning Customer Address data")
# data3['state'].replace('New South Wales', 'NSW', inplace=True)
# data3['state'].replace('Victoria', 'VIC', inplace=True)
# data3_clean = data3.dropna()
# # print(data3_clean.head())
# # print("Merging the Customer Address with Customer Demographic data")
# demo_addr_df = pd.merge(data2_clean, data3_clean, left_index=True, right_index=True)
# demo_addr_df = demo_addr_df.dropna()
# # print(demo_addr_df.head())
# # print("Cleaning Transaction Data")
# data = data.sort_values('customer_id')
# # print(data.head())
# # print("Converting the data into the correct format")
# data['transaction_date'] = pd.TimedeltaIndex(data['transaction_date'], unit='d') + dt.datetime(1900, 1, 1)
# data['product_first_sold_date'] = pd.TimedeltaIndex(data['product_first_sold_date'], unit='d') + dt.datetime(1900, 1, 1)
# data_clean = data.dropna()  # Dropping empty rows
# # data.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_data_update.xlsx")
# print(data_clean.shape)
# print(data.head())
# # print()
# # print(data_clean['transaction_date'].describe(datetime_is_numeric=True))
# # print("Generating a profit and last purchase days ago")
# most_recent_purchase = data_clean['transaction_date'].max()
# data_clean['last_purchase_days_ago'] = most_recent_purchase - data_clean['transaction_date']
# data_clean['last_purchase_days_ago'] /= np.timedelta64(1, 'D')
# data_clean['profit'] = data_clean['list_price'] - data_clean['standard_cost']
# # print(data_clean.head())
# print(data_clean[data_clean['last_purchase_days_ago'] > 365].shape)
# print()
# print("Generating the RFM table")
# rfmTable = data_clean.groupby('customer_id').agg({
#     'last_purchase_days_ago': lambda x: x.min(),
#     'customer_id': lambda x: len(x),
#     'profit': lambda x: x.sum()
# })
#
# rfmTable.rename(columns={
#     'last_purchase_days_ago': 'recency',
#     'customer_id': 'frequency',
#     'profit': 'monetary_value'
# }, inplace=True
# )
# # print(rfmTable.head())
# # print(rfmTable.shape)
# # print("Creating a quantile range")
# quartiles = rfmTable.quantile(q=[0.25, 0.50, 0.75])
# # print(quartiles)
# #
# # print("Calculating the RFM scores")
# #
# #
# def ROneHotEncoder(x, p, d):
#     if x <= d[p][0.25]:
#         return 1
#     elif x <= d[p][0.5]:
#         return 2
#     elif x <= d[p][0.75]:
#         return 3
#     else:
#         return 4
# #
# #
# # # descending order
# def FMOneHotEncoder(x, p, d):
#     if x <= d[p][0.25]:
#         return 4
#     elif x <= d[p][0.5]:
#         return 2
#     elif x <= d[p][0.75]:
#         return 3
#     else:
#         return 1
#
# #
# rfmSeg = rfmTable
# rfmSeg['r_score'] = rfmSeg['recency'].apply(ROneHotEncoder, args=('recency', quartiles))
# rfmSeg['f_score'] = rfmSeg['frequency'].apply(FMOneHotEncoder, args=('frequency', quartiles))
# rfmSeg['m_score'] = rfmSeg['monetary_value'].apply(FMOneHotEncoder, args=('monetary_value', quartiles))
# print(rfmSeg.head())
# # print("Generating the RFM class and Total scores")
# rfmSeg['rfm_class'] = 100 * rfmSeg['r_score'] + 10 * rfmSeg['f_score'] + rfmSeg['m_score']
# rfmSeg['total_score'] = rfmSeg['r_score'] + rfmSeg['f_score'] + rfmSeg['m_score']
# # print(rfmSeg.head())
# # print("Getting the RFM class min, median,max and their quantiles")
# rfm_quartiles = (rfmSeg['rfm_class'].min(), rfmSeg['rfm_class'].quantile(q=0.25),
#                  rfmSeg['rfm_class'].median(), rfmSeg['rfm_class'].quantile(q=0.75),
#                  rfmSeg['rfm_class'].max())
# # print(rfm_quartiles)
# # print("Assigning Title to the classes")
# #
# #
# def RFMClassOneHotEncoder(x, p, d):
#     if x <= d[0]:
#         return 'gold'
#     elif x <= d[1]:
#         return 'silver'
#     elif x <= d[2]:
#         return 'bronze'
#     else:
#         return 'basic'
# #
# #
# rfmSeg['customer_title'] = rfmSeg['rfm_class'].apply(RFMClassOneHotEncoder, args=('rfm_class', rfm_quartiles))
# print(rfmSeg)
# # rfmSeg.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_RFM_data.xlsx")
# #
# addr_demo_txns = pd.merge(rfmSeg, demo_addr_df, left_index=True, right_index=True)
# # addr_demo_txns.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_combined_data.xlsx")
# # print(addr_demo_txns.head())
# print(addr_demo_txns.columns)
# # print(addr_demo_txns.shape)

# sns.catplot(x='age',y='customer_title',data= addr_demo_txns, kind= 'bar', orient= 'h',ci=False)
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_customer_title2.svg")
# sns.catplot(x='age',y='customer_title',data= addr_demo_txns, kind= 'bar', orient= 'h',ci=False, hue='gender')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_customer_2.png")
# sns.catplot(x='past_3_years_bike_related_purchases',y='customer_title',data= addr_demo_txns, kind= 'bar', orient= 'h',ci=False, hue= 'job_industry_category')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_customer_3.png")
# pl= sns.regplot(x='recency', y='monetary_value', data=addr_demo_txns, ci=False, fit_reg=False)
# pl.set_title('Recency against profit')
# plt.scatter(addr_demo_txns['recency'], addr_demo_txns['monetary_value'])
# data5 = pd.read_excel(
#     "C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_combined_data.xlsx",
#     index_col='customer_id')
# sns.catplot(x='monetary_value',y='age_class',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target3.png")

# sns.catplot(x='monetary_value',y='rfm_class',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target4.png")

# sns.catplot(x='monetary_value',y='gender',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target5.png")

# sns.catplot(x='monetary_value',y='frequency',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target6.png")

# sns.catplot(x='monetary_value',y='wealth_segment',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target7.png")

# sns.catplot(x='past_3_years_bike_related_purchases',y='job_industry_category',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target9.png")

# sns.catplot(x='past_3_years_bike_related_purchases',y='job_industry_category',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target9.png")

# addr_demo_txns['customer_segment'] = addr_demo_txns['total_score'].map({
#     3: 'Platinum',
#     4: 'Very Loyal',
#     5: 'Becoming Loyal',
#     6: 'Recent',
#     7: 'Average',
#     8: 'High Risk',
#     9: 'Evasive',
#     10: 'Losing',
#     11: 'Inactive',
#     12: 'Lost'
# })
# print(addr_demo_txns.head())
"""To generate the top 1000 after sorting"""
# top_1000 = addr_demo_txns.sort_values('rfm_class').head(1000)
# print(top_1000)

"""generating a pie chart for top 1000 customer segments"""
# top_1000.customer_segment.value_counts().plot.pie(autopct=lambda pct: str(round(pct, 2)) + '%')
# plt.title('Distribution of customers segments')
# plt.ylabel('')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_customer_segment_pie.png")

""" To get the counts of each value of each customer title"""
# c = addr_demo_txns.groupby('customer_title').agg({'age': lambda x: len(x)})
# c.rename(columns={'age':'count'},inplace=True)
# c['count'] = c['count'].astype(int)
# print(c)
""" To get the counts of each value of total RFM scores"""
# w = addr_demo_txns.groupby(['total_score']).agg({'age': lambda x: x.count()}).cumsum()
# w.rename(columns={'age':'count'},inplace=True)
# w['count'] = w['count'].astype(int)
# print(w)
"""To get the counts of each value of total score RFM for top 1000"""
# m = top_1000.groupby(['total_score']).agg({'age': lambda x: len(x)})
# m.rename(columns={'age':'count'},inplace=True)
# m['count'] = m['count'].astype(int)
# print(m)
"""state and customer segment"""
# sns.catplot(x='state',y='customer_segment',data= top_1000, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target_state_seg.png")

# """ To get the counts of each value of states with the titles"""
# s = addr_demo_txns.groupby(['state','customer_title']).agg({'age': lambda x: len(x)})
# s.rename(columns={'age':'count'},inplace=True)
# s['count'] = s['count'].astype(int)
# print(s)
# top_1000.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_top_1000_data.xlsx")

# sns.catplot(x='monetary_value',y='wealth_segment',data= top_1000, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target_top.png")

# sns.catplot(x='owns_car',y='state',data= top_1000, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target_state.png")

# sns.catplot(x='owns_car',y='job_industry_category',data= top_1000, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target_ind_car.png")

# sns.catplot(x='age_class',y='customer_segment',data= top_1000, kind= 'bar', orient= 'h',ci=False, hue= 'gender')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target_cus_age.png")

# sns.catplot(x='total_score',y='customer_segment',data= addr_demo_txns, kind= 'bar', orient= 'h',ci=False, hue=['r_score', 'f_score','m_score'] )
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target_cus_rfm.png")

# sns.catplot(x='monetary_value',y='job_industry_category',data= data5, kind= 'bar', orient= 'h',ci=False, hue= 'customer_title')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_target8.png")



# addr_demo_txns.customer_title.value_counts().plot.pie(autopct=lambda pct: str(round(pct, 2)) + '%')
# plt.title('Distribution of customers')
# plt.ylabel('')
# plt.savefig("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/KPMG_customer_pie.svg")

# #
# # writer = pd.ExcelWriter('kpmg_cleaned_data.xlsx',
# #                         engine = 'xlsxwriter')
# # addr_demo_txns.to_excel(writer, sheet_name = 'Dataset')
# # top_1000.to_excel(writer, sheet_name = 'Top 1000')
# # writer.save()
# # writer.close()
