# """1.1"""
# speed_limit = int(input())
# driver_speed =int(input())
#
# if driver_speed < speed_limit:
#     print("Driver's speed is within the legal limit.")
# else:
#     X = driver_speed - speed_limit
#     if X < 10:
#         print("Driver is speeding! clocked at ", X ,"mph over the limit. Issue a warning")
#     elif X > 9 and X < 20:
#         print("Driver is speeding! clocked at ", X ,"mph over the limit. Issue a ticket with a $50 fine.")
#     elif X >19 and X < 30:
#         print("Driver is speeding! clocked at ", X ,"mph over the limit. Issue a ticket with a $75 fine.")
#     else:
#         print("Driver is speeding! clocked at ", X ,"mph over the limit. Issue a ticket with a $100 fine.")
#
# """1.2"""
# name = input()
# time = int(input())
# if name == "Facebook":
#     print("All employees 4 weeks of vacation")
# elif name == "Amazon":
#     if time <= 1:
#         print("New employees 2 weeks of vacation")
#     else:
#         print("3 weeks of vacation")
# else:
#     if time < 3:
#         print("New employees 3 weeks of vacation")
#     elif time >= 3 and time < 5:
#         print("4 weeks of vacation")
#     else:
#         print("5 weeks of vacation")
#
# """1.3"""
# string = input()
# charc = input()
# len_s = len(string)
# dot = ""
# for index in range(0,len_s):
#     cha = string[index]
#     if string[index] == charc:
#         dot = dot + str(index)
#     else:
#         dot += cha
# print(dot)
#
# """1.4"""
# word_1 = input()
# word_2 = input()
#
# res = ""
# for index in range(0,len(word_1)):
#     res += word_1[index] + word_2[index]
# print(res)
#
# """1.5"""
# num = 60 # or the random values pre-set
# num1 = int(input())
#
# if num1 > num:
#     X = num1 - num
#     if X > 50:
#         print("Cold")
#     elif X >= 21 and X <= 50:
#         print("Warmish")
#     elif X >= 6 and X <= 20:
#         print("warm")
#     elif X >= 1 and X <= 5:
#         print("hot")
#     else:
#         X = 0
#         print("YOU WIN")
#         num1 += num

"""class 1 june"""
# game_age_dict = {"danny":[24,"F"],
#                  "michael":[38,"M"],
#                  "Emma":[22,"F"],
#                  "mackay":[15,"M"],
#                  "bright":[12,"M"],
#                  "Yetunde":[18,"F"],
#                  "blessing":[20,"F"]}
# print(game_age_dict )
# game_age_dict = {i.lower(): v for i, v in game_age_dict.items()}
# print(game_age_dict)
# lst = [10,3,4,6]
# print(sorted(lst))
# sorted_dict = sorted(game_age_dict.items(), key = lambda x:x[0])
# print(sorted_dict)
#
# # Given an array of integers nums and an integer target,
# # return indices of the two numbers such that they add up to target.
# nums = [7, 3, 4, 4, 2]
# target = 9
# for i in range(len(nums)):
#     for j in range(i+1,len(nums)):
#         if nums[i] + nums[j] == target:
#             print(i,j)
#
# print()
# print("using dictionary")
# myDict = {}
# for index, value in enumerate(nums):
#     diff = target - value
#     if diff in myDict:
#         print(myDict[diff], index)
#     myDict[value] = index


# res = []
# for sub in test_list:
#     for word in sub.split():
#
#         # check for keyword using iskeyword()
#         if keyword.iskeyword(word):
#             res.append(word)


# Python3 code to demonstrate working of
# Extract Keywords from String List

# # Using iskeyword() + loop + split()
# import keyword
#
# # initializing list
# test_list = ["Gfg is True", "Gfg will yield a return",
# 			"Its a global win", "try Gfg"]
#
# # printing original list
# print("The original list is : " + str(test_list))
#
#
# # iterating using loop
# res = []
# for sub in test_list:
# 	for word in sub.split():
#
# 	# check for keyword using iskeyword()
# 		if keyword.iskeyword(word):
# 			# print(word)
# 			res.append(word)
#
# # printing result
# print(sub)
# print("Extracted Keywords : " + str(res))


# Python3 code to demonstrate working of
# Get values of particular key in list of dictionaries
# Using list comprehension

# initializing list
# test_list = [{'gfg' : 1, 'is' : 2, 'good' : 3},
# 			{'gfg' : 2}, {'best' : 3, 'gfg' : 4}]
#
# # printing original list
# print("The original list is : " + str(test_list))
#
#
# # Using list comprehension
# # Get values of particular key in list of dictionaries
# res = [ sub['gfg'] for sub in test_list ]
#
# # printing result
# print("The values corresponding to key : " + str(res))
# import pandas as pd
# with open("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Project/api.txt") as top:
#     df = top.read()
# print(df.groupby('Code').agg(lambda g: list(g.dropna())).to_dict(orient='index'))
# import collections
# result = collections.defaultdict(list)
# for vpc in tags:
#     result[vpc['Code']].append(vpc['Name'])
#     print(result)
# print({tag['Code']:tag['Name'] for tag in tags})
# # df1 = pd.DataFrame([a])
# print(df1)
# #
# results = df1.to_dict(orient='list')
# print(results)

# file_handle = open("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Project/divine_freedom.txt")
# file = file_handle.read()
# line = file.rstrip()
# word_lst = []
# word_dict = {}
# words = line.split()
#
# for word in words:
#     if word.startswith('('):
#         word_lst.append(word[1:])
# for country_code in word_lst:
#     if country_code == '':
#         continue
#     word_dict[country_code] = word_dict.get(country_code, 0)+1
# print(sorted(word_dict.items()))

#
import urllib.request as request
import json

with request.urlopen(
        'https://pkgstore.datahub.io/core/country-list/data_json/data/8c458f2d15d9f2119654b29ede6e45b8/data_json.json') as response:
    # if response.getcode() == 200:
        source = response.read()
        country_name = json.loads(source)
    # else:
        # print('An error occurred while attempting to retrieve data from the API.')
# print(country_name)
with open("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Project/divine_freedom.txt") as top:
    divine = top.read()
# print(divine)
line = divine.strip()
words = line.split()
# print(words)
word_lst = []
word_dict = {}
for word in words:
    if word.startswith('('):
        word_lst.append(word[1:])
for country_code in word_lst:
    if country_code == '':
        continue
    word_dict[country_code] = word_dict.get(country_code, 0) + 1
# print(word_dict)
def country_code(dic):
    code_dict = {}
    for i in country_name:
        for k in i:
            code_dict[i["Code"]] = i["Name"]
    """assigning the values of the two dictionaries to each other"""
    trans_dict = {}
    for j in dic:
        for key in code_dict:
            if j == key:
                trans_dict[code_dict[key]] = dic[j]
    """reordering the dictionary to return in a descending order"""
    import operator
    res = dict(sorted(trans_dict.items(), key=operator.itemgetter(1), reverse=True))
    """returning the top 20 countries"""
    top_country = dict(list(res.items())[:20])
    return top_country
print(country_code(word_dict))

"""creating a bar char of country name and code"""
import matplotlib.pyplot as plt
data = country_code(word_dict)
x = list(data.keys())
values = data.values()
ax =plt.bar(x, values,tick_label=x)
plt.xlabel('x', fontsize=15)
plt.xticks(fontsize=10, rotation= 'vertical')
plt.tick_params(axis='x', labelsize=14)
plt.title('Country code and count')
plt.show()
#
