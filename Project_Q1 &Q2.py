# import urllib.request as request
# import json
# """import of the json file"""
# with request.urlopen(
#         'https://pkgstore.datahub.io/core/country-list/data_json/data/8c458f2d15d9f2119654b29ede6e45b8/data_json.json') as response:
#     if response.getcode() == 200:
#         source = response.read()
#         country_name = json.loads(source)
#     else:
#         print('An error occurred while attempting to retrieve data from the API.')
#
# """import of the text file"""
# with open("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Project/divine_freedom.txt") as top:
#     divine = top.read()
# """striping and splitting of the divine data to easy accessibility"""
# line = divine.strip()
# words = line.split()
# word_lst = []
# word_dict = {}
# """extraction of the country code from the data"""
# for word in words:
#     if word.startswith('('):
#         word_lst.append(word[1:])
# """counting of the country codes"""
# for country_code in word_lst:
#     if country_code == '':
#         continue
#     word_dict[country_code] = word_dict.get(country_code, 0) + 1
# """converting the json data into one single dictionary"""
# def country_code(dic):
#     code_dict = {}
#     for i in country_name:
#         for k in i:
#             code_dict[i["Code"]] = i["Name"]
#     """assigning the values of the two dictionaries to each other"""
#     trans_dict = {}
#     for j in dic:
#         for key in code_dict:
#             if j == key:
#                 trans_dict[code_dict[key]] = dic[j]
#     """reordering the dictionary to return in a descending order"""
#     import operator
#     res = dict(sorted(trans_dict.items(), key=operator.itemgetter(1), reverse=True))
#     """returning the top 20 countries"""
#     top_country = dict(list(res.items())[:20])
#     return top_country
# print(country_code(word_dict))
#
# """creating a bar char of country name and code"""
# import matplotlib.pyplot as plt
# data = country_code(word_dict)
# x = list(data.keys())
# values = data.values()
# ax =plt.bar(x, values,tick_label=x)
# plt.xlabel('x', fontsize=15)
# plt.xticks(fontsize=10, rotation= 'vertical')
# plt.tick_params(axis='x', labelsize=14)
# plt.title('Country code and count')
# plt.show()

rollNumbers = [122, 233, 353, 456]
names = ['alex', 'bob', 'can', 'don']
res = {i: j for (i, j) in zip(rollNumbers, names)}
print(res)
# for i in rollNumbers:
#     for j in names:
#         res = zip(rollNumbers,names)
#     print(res)

sentence = 'I love programming in Python. How about you?'
sent = sentence[::-1]
print("The reverse of the sentences is: ", sent)

"""The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number 
is the sum of the two preceding ones, starting from 0 and 1. """


def fibonacci(n):
    if n <= 1:
        return n
    if n == 2:
        return 2
    current = 0
    previous1 = 1
    previous2 = 1
    for i in range(3, n + 1):
        current = previous1 + previous2
        previous2 = previous1
        previous1 = current
    return current


# test case
fibonacci(n=3)  # 2

"""or using recursion"""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# test case
fibonacci(n=3)  # 2
