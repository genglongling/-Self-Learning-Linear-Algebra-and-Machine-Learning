url='http://search.dangdang.com/?key=python&act=input'
headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
from lxml import html
import requests
from matplotlib import pyplot as plt
response = requests.get(url, headers=headers) # 200 ok, 4xx客户端， 5xx
print(response.status_code)
data = response.text
selector = html.fromstring(data)
ul_list= selector.xpath('//div[@id="search_nature_rg"]/ul/li')
book_list = []
for li in ul_list[:10]:
    name = li.xpath('p[1]/a/@title')[0]
    url = li.xpath('a/img/@data-original')
    if len(url)==0:
        url = li.xpath('a/img/@src')

    url=url[0]
    if str(url).startswith('//'):
        url=str(url).replace('//','')
    url="http://"+url
    print(url)

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        img_Data = response.content #二进制
        with open('./images/'+name+'.jpg', mode='wb') as f:
            f.write(img_Data)
    #     #plt.imshow(img_Data)

