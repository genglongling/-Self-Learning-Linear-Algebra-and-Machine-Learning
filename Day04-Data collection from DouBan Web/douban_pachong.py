# 豆瓣爬虫
# 爬虫：1.封面 2.电影名 3.类型 4.上映地址 5.想看人数
# 降序排序
# 存储：为csv
# 绘制这些电影类型占比图
# https://movie.douban.com/cinema/later/shenyang/
import requests
from lxml import html
from matplotlib import pyplot as plt
import numpy as np

url='https://movie.douban.com/cinema/later/shenyang/'
headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
response = requests.get(url, headers=headers) # 200 ok, 4xx客户端， 5xx
#print(response.status_code)
data = response.text
#print(data)

selector = html.fromstring(data)
ul_list= selector.xpath('//div[@id="showing-soon"]/div')
film_list=[]

# print(ul_list)
for li in ul_list:
    film_dict={}
    name = li.xpath('div[@class="intro"]/h3/a/text()')[0]
    # print(name)

    # make dir
    img = li.xpath('a/img/@src')[0]
    # print(img)
    response = requests.get(img, headers=headers)
    if response.status_code == 200:
        img_Data = response.content #二进制
        with open('./images/'+name+'.jpg', mode='wb') as f:
            f.write(img_Data)

    type = li.xpath('div[@class="intro"]/ul/li[2]/text()')[0]
    place = li.xpath('div[@class="intro"]/ul/li[3]/text()')[0]
    num = li.xpath('div[@class="intro"]/ul/li[@class="dt last"]/span/text()')[0]
    num = str(num)[:len(num)-3]
    # print(type)
    # print(place)
    # print(num)

    film_dict['name']=name
    film_dict['img'] = img
    film_dict['type'] = type
    film_dict['place'] = place
    film_dict['num'] = int(num) # for sorting
    film_list.append(film_dict)


film_list.sort(key=lambda dict:dict['num'], reverse=True)
print(film_list)

import pandas as pd
df=pd.DataFrame(film_list)
df.to_csv("Douban_statistics.csv", index=False)

x = np.random.randint(3000,500000,10)
plt.figure(figsize=(10,8))
import string
type_list=[]
freq_list=[]

for li in film_list:
    if li['type'] in type_list:
        i=0
        for type in type_list:
            if type==li['type']:
                break
            i = i + 1
        freq_list[i]=freq_list[i]+1
    else:
        type_list.append(li['type'])
        freq_list.append(1)

print(type_list)
print(freq_list)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
names = [f'{v}' for v in type_list]
explodes=[0.0 for _ in range(len(names))]
explodes[0]=0.3
plt.pie(freq_list, labels=names, explode=explodes, shadow=True, autopct='%1.2f%%')
plt.legend()
plt.savefig('img_by_type.png')
plt.show() #结束