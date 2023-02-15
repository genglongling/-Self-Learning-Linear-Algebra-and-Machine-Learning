#地区，本土确诊，无症状
url='https://api.inews.qq.com/newsqa/v1/query/inner/publish/modules/list?modules=localCityNCOVDataList,diseaseh5Shelf'
headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
from lxml import html
import requests
from matplotlib import pyplot as plt
response = requests.get(url, headers=headers) # 200 ok, 4xx客户端， 5xx
print(response.status_code)
data = response.content.decode() #二进制解码
#print(data)
import json
data = json.loads(data)
#print(data) #str to dict

info_list=[]
for key, val in data.items():
    if key=="data":
        info_list = val['localCityNCOVDataList']
        break

#print(info_list)

result=[]
for info in info_list:
    prepated_dict = {}
    prepated_dict['city']=info['city']
    prepated_dict['local_confirm_add'] = info['local_confirm_add']
    prepated_dict['local_wzz_add'] = info['local_wzz_add']
    result.append(prepated_dict)

print(result)

# 异步加载，一般在诊断->网络->Fetch/XHR
