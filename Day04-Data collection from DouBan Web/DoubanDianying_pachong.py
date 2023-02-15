import requests
from lxml import html
film_list=[]

iteration=0
for iteration in range(15):
    url = 'https://movie.douban.com/j/search_subjects?type=tv&tag=%E7%83%AD%E9%97%A8&sort=recommend&page_limit=20&page_start='+str(iteration)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)  # 200 ok, 4xx客户端， 5xx
    print(response.status_code)
    data = response.content.decode()  # 二进制解码
    # print(data)
    import json

    data = json.loads(data)  # str to dict
    data = data["subjects"]
    print(data)  # dict to list

    for dic in data:
        film_dict = {}
        film_dict['name'] = dic['title']
        if dic['rate'] == "":
            film_dict['rate'] = 0
        else:
            film_dict['rate'] = dic['rate']
        film_list.append(film_dict)
    iteration=iteration+20

print(film_list)
print(len(film_list))

import pandas as pd
df=pd.DataFrame(film_list)
df.to_csv("DoubanDianying_statistics.csv", index=False)