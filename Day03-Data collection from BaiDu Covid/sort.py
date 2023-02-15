import string
from random import randint
data = {f'学生{v}': randint(10, 100) for v in string.ascii_uppercase[:7]}
print(data)
# 转成列表
data = list(data.items())
def sort_by_score(tup):
    return tup[1]
# data.sort(key=sort_by_score)
data.sort(key=lambda tup:tup[1], reverse=True)
print(data)