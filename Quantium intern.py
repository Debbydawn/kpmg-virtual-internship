import xlrd
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from string import punctuation

transaction_data = pd.read_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/QVI_transaction_data.xlsx")

behaviour_data = pd.read_csv("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/QVI_purchase_behaviour.csv")
#
# print(transaction_data.info())
# print(behaviour_data.info())
#
transaction_data['DATE'] = pd.to_datetime(transaction_data['DATE'], errors='coerce', unit='d', origin='1900-01-01')
# print(transaction_data['DATE'].head())

#
# print(transaction_data.info())
# print(behaviour_data.info())

# def duplicate_in_columns(df):
#     for col in df.columns:
#         dup = df.duplicated().any()
#         print("{} has {} duplicated value(s)".format(col, dup))
#
#
# print(duplicate_in_columns(transaction_data))
# print(duplicate_in_columns(behaviour_data))


# # """Return completeness percentage"""
# def completeness_percent(df):
#     completion = df.isna().sum().sum() / df.size * 100
#     return "The data has a completion percentage of", completion
#
#
# print(completeness_percent(transaction_data))
# print(completeness_percent(behaviour_data))


# """check for the uniqueness of each columns"""
# def unique_col(df):
#     for col in df.columns:
#         uni = df.nunique()
#     return uni
#
#
# print(unique_col(transaction_data))
# print(unique_col(behaviour_data))

# print()
# # def value(df):
# #     for col in df.columns:
# #         count= df[col].value_counts()
# #     return count
# # print(value(data))
#
# print(transaction_data['DATE'].value_counts())
# print(transaction_data['STORE_NBR'].value_counts())
# print(transaction_data['LYLTY_CARD_NBR'].value_counts())
# print(transaction_data['TXN_ID'].value_counts())
# print(transaction_data['PROD_NBR'].value_counts())
# print(transaction_data['PROD_NAME'].value_counts())
# print(transaction_data['PROD_QTY'].value_counts())
# print(transaction_data['TOT_SALES'].value_counts())

"""Cleaning the DATA"""
"""removing products with the name Salsa"""
chips = transaction_data[transaction_data['PROD_NAME'].str.contains('Salsa') == False].copy()

# print(chips['PROD_NAME'].head(28))
"""separating the product name column to extract items"""


def remove_punc_digit_no_g(val):
    val = [v for v in val if v not in punctuation]
    result = ''.join([i for i in val if not i.isdigit()])
    final = ''.join([e for e in result if e.isalnum() or e == ' '])
    return final.replace('g', '')


chips['PROD_NAME'].apply(lambda x: remove_punc_digit_no_g(x)).str.split(expand=True).stack().value_counts()
# print(chips['PROD_NAME'].head(28))


"""removes the digits"""


def remove_punc_digit(val):
    val = [v for v in val if v not in punctuation]
    result = ''.join([i for i in val if not i.isdigit()])
    final = ''.join([e for e in result if e.isalnum() or e == ' '])
    return final


chips['PROD_NAME'] = chips['PROD_NAME'].apply(lambda x: remove_punc_digit(x))

# print(chips['PROD_NAME'].head(28))

chips[chips['PROD_QTY'] == 200]
chips[chips['LYLTY_CARD_NBR'] == 226000]
final = chips[chips['LYLTY_CARD_NBR'] != 226000]

bydate = final.groupby('DATE')[['TXN_ID']].count().reset_index()
# print(bydate)
"""creating a plot for transaction trend"""
# timeline = bydate.DATE
# graph = bydate['TXN_ID']
#
# fig, ax = plt.subplots(figsize = (10, 5))
# ax.plot(timeline, graph)
#
# date_form = DateFormatter("%Y-%m")
# ax.xaxis.set_major_formatter(date_form)
# plt.title('Transaction over time')
# plt.xlabel('Time')
# plt.ylabel('Total Sales')
#
# plt.show()
# c_december = c[(c.index < "2019-01-01") & (c.index > "2018-11-30")]
december = bydate[(bydate.DATE < "2019-01-01") & (bydate.DATE > "2018-11-30")]


# time = december.DATE
# graph = december['TXN_ID']
# # print(december)
# plt.figure(figsize=(10, 5))
# plt.plot(time, graph)
# plt.xlabel('Date')
# plt.ylabel('Total Sales')
# plt.title('Total Sales in December')
#
# plt.show()


#
def pack_size(inp):
    inp = ''.join([x for x in inp if x.isdigit()])
    return str(inp + 'g')


transaction_data['Pack Size'] = transaction_data['PROD_NAME'].apply(lambda d: pack_size(d))
transaction_data['Pack Size'].value_counts()
# print(transaction_data.head(5))

december.reset_index(drop=True, inplace=True)
# print(december.head())

# december['DATE'] = december.index + 1
# december.head()
# plt.figure(figsize = (10,5))
# sns.barplot(x = 'DATE', y ='TXN_ID', data = december)
#
# plt.show()

#
transaction_data['Brand Name'] = transaction_data['PROD_NAME'].str.split().apply(lambda x: x[0])
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Red', 'RRD')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('SNBTS', 'SUNBITES')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('infzns', 'Infuzions')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('WW', 'woolworths')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Smith', 'Smiths')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('NCC', 'Natural')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Dorito', 'Doritos')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Grain', 'GrnWves')
# print(transaction_data['Brand Name'].value_counts())


behaviour_data['LIFESTAGE'].value_counts()
behaviour_data['PREMIUM_CUSTOMER'].value_counts()
merged = pd.merge(behaviour_data, transaction_data, on='LYLTY_CARD_NBR')
# print(merged.isnull().sum())
#  print(merged)
# merged.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/QVI_merged_2.xlsx")


# byc = merged.groupby(['LIFESTAGE','PREMIUM_CUSTOMER'])[['TOT_SALES']].sum().reset_index()
#
# fig = px.bar(byc,byc['LIFESTAGE'],byc['TOT_SALES'],byc['PREMIUM_CUSTOMER'],text=(byc['TOT_SALES']))
# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
# fig.update_layout(title='Proportion of Sales',title_x=0.5)
#
# fig.show()


#
# byc = merged.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])[['LYLTY_CARD_NBR']].count().reset_index()
#
# fig = px.bar(byc, byc['LIFESTAGE'], byc['LYLTY_CARD_NBR'].unique(), byc['PREMIUM_CUSTOMER'],
#              text=byc['LYLTY_CARD_NBR'].unique())
# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
# fig.update_layout(title='Proportion of Customers', title_x=0.5)
# fig.show()

# unique_cust = merged[merged['LYLTY_CARD_NBR'].isin(list(pd.unique(merged['LYLTY_CARD_NBR'])))].copy()
# byc = unique_cust.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])[['PROD_QTY', 'LYLTY_CARD_NBR']].sum().reset_index()
# byc['AVG'] = byc['PROD_QTY'] / byc['LYLTY_CARD_NBR']
#
# fig = px.bar(byc, byc['LIFESTAGE'], byc['AVG'], byc['PREMIUM_CUSTOMER'])
# # Change the bar mode
# fig.update_layout(barmode='group', title='Average Units per customer', title_x=0.5)
# fig.show()
#
# byc = merged.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])[['TOT_SALES', 'PROD_QTY']].sum().reset_index()
# byc['AVG'] = byc['TOT_SALES'] / byc['PROD_QTY']
#
# fig = px.bar(byc, byc['LIFESTAGE'], byc['AVG'], byc['PREMIUM_CUSTOMER'])
# fig.update_layout(barmode='group', title='Average Sales per Unit', title_x=0.5)
# fig.show()
#
import scipy.stats

merged['PricePerUnit'] = merged['TOT_SALES'] / merged['PROD_QTY']

target = merged[(merged['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"])) & (
        merged['PREMIUM_CUSTOMER'] == 'Mainstream')]
non_target = merged[(merged['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"])) & (
        merged['PREMIUM_CUSTOMER'] != 'Mainstream')]

# print(target)
# #
# target.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/QVI_target_.xlsx")

# # non_target.to_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/QVI_filtered_.xlsx")
#
# print(scipy.stats.ttest_ind(target['PricePerUnit'], non_target['PricePerUnit']))


"""online"""
# sns.countplot(y = behaviour_data['LIFESTAGE'], order = behaviour_data['LIFESTAGE'].value_counts().index)
#
# plt.show()

# plt.figure(figsize = (12, 7))
# sns.countplot(y = behaviour_data['PREMIUM_CUSTOMER'], order = behaviour_data['PREMIUM_CUSTOMER'].value_counts().index)
# plt.xlabel('Number of Customers')
# plt.ylabel('Premium Customer')
#
# plt.show()


# plt.figure(figsize=(10, 5))
# plt.hist(target, label='Mainstream')
# plt.hist(non_target, label='Premium & Budget')
# plt.legend()
# plt.xlabel('Price per Unit')
#
# plt.show()

targetBrand = target.loc[:, ['Brand Name', 'PROD_QTY']]
targetSum = targetBrand['PROD_QTY'].sum()
targetBrand['Target Brand Affinity'] = targetBrand['PROD_QTY'] / targetSum
targetBrand = pd.DataFrame(targetBrand.groupby('Brand Name')['Target Brand Affinity'].sum())

# Non-target segment
nonTargetBrand = non_target.loc[:, ['Brand Name', 'PROD_QTY']]
nonTargetSum = nonTargetBrand['PROD_QTY'].sum()
nonTargetBrand['Non-Target Brand Affinity'] = nonTargetBrand['PROD_QTY'] / nonTargetSum
nonTargetBrand = pd.DataFrame(nonTargetBrand.groupby('Brand Name')['Non-Target Brand Affinity'].sum())

# Merge the two dataframes together

brand_proportions = pd.merge(targetBrand, nonTargetBrand, left_index = True, right_index = True)
print(brand_proportions.head())
#
brand_proportions['Affinity to Brand'] = brand_proportions['Target Brand Affinity'] / brand_proportions['Non-Target ' \
                                                                                                        'Brand ' \
                                                                                                        'Affinity']
print(brand_proportions.sort_values(by = 'Affinity to Brand', ascending = False))

targetSize = target.loc[:, ['Pack Size', 'PROD_QTY']]
targetSum = targetSize['PROD_QTY'].sum()
targetSize['Target Pack Affinity'] = targetSize['PROD_QTY'] / targetSum
targetSize = pd.DataFrame(targetSize.groupby('Pack Size')['Target Pack Affinity'].sum())

# Non-target segment
nonTargetSize = non_target.loc[:, ['Pack Size', 'PROD_QTY']]
nonTargetSum = nonTargetSize['PROD_QTY'].sum()
nonTargetSize['Non-Target Pack Affinity'] = nonTargetSize['PROD_QTY'] / nonTargetSum
nonTargetSize = pd.DataFrame(nonTargetSize.groupby('Pack Size')['Non-Target Pack Affinity'].sum())

# Merge the two dataframes together

pack_proportions = pd.merge(targetSize, nonTargetSize, left_index = True, right_index = True)
print(pack_proportions.head())

pack_proportions['Affinity to Pack'] = pack_proportions['Target Pack Affinity'] / pack_proportions['Non-Target Pack Affinity']
pack_proportions.sort_values(by = 'Affinity to Pack', ascending = False)

#It looks like mainstream singles/couples are more likely to purchase a 270g pack size compared to other pack sizes.
# Which brand offers 270g pack size?

print(merged.loc[merged['Pack Size'] == 270, :].head(10))
print(merged.loc[merged['Pack Size'] == 270, 'Brand Name'].unique())
#
# Twisties is the only brand that offers 270g pack size.
#
# Conclusion Sales are highest for (Budget, OLDER FAMILIES), (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream,
# RETIREES) We found that (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream, RETIREES) are mainly due to the fact
# that there are more customers in these segments (Mainstream, YOUNG SINGLES/COUPLES) are more likely to pay more per
# packet of chips than their premium and budget counterparts They are also more likely to purchase 'Tyrrells' and
# '270g' pack sizes than the rest of the population
