import requests
from lxml import html
url='https://scholar.google.com/scholar?as_ylo=2022&q=backdoor+attacks+on+graphs&hl=zh-CN&as_sdt=0,5'
#url='http://www.zhihu.com'
# 以浏览器形式访问
headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
response = requests.get(url, headers=headers) # 200 ok, 4xx客户端， 5xx
print(response.status_code)
data = response.text
print(data)
# with open('dangdang_website.html', mode='w') as f:
#     f.write(data)
# utf-8转换编码方式
selector = html.fromstring(data)
# xpath语法用于提取
# //代表任意位置
# //标签名[@属性=属性值] id/class
# 获取标签中属性值
# names=selector.xpath('//div[@id="search_nature_rg"]/ul/li/p[1]/a/@title')
# print(names)
# # 获取标签中内容 /text()
# prices=selector.xpath('//div[@id="search_nature_rg"]/ul/li/p[@class="price"]/span[1]/text()')
# print(prices)
# comments=selector.xpath('//div[@id="search_nature_rg"]/ul/li/p[@class="search_star_line"]/a[@class="search_comment_num"]/text()')
# print(comments)
ul_list= selector.xpath('//div[@id="search_nature_rg"]/ul/li')
book_list = []
for li in ul_list:
    name = li.xpath('p[1]/a/@title')[0]
    handling_price=li.xpath('p[@class="price"]/span[1]/text()')[0]
    price=float(handling_price[1:])
    handling_comment=li.xpath('p[@class="search_star_line"]/a[@class="search_comment_num"]/text()')
    comment=0 #可能为空
    # if handling_comment!=[]:
    #     temp=handling_comment[0]
    #     temp=temp.replace('条','')
    #     temp = temp.replace('评', '')
    #     temp = temp.replace('论', '')
    #     comment=int(temp)
    handling_comment='0条评论' if len(handling_comment)==0 else handling_comment[0]
    comment=int(handling_comment.replace("条评论",''))
    # print(name)
    # print(price)
    # print(comment)
    book_list.append({
        "name":name,
        "price":price,
        "comment":comment
    })

# for book in book_list:
#     print(book)

#排序算法
# python sort()
# 数字，直接sort（）
#data.sort(key=sort_by_score)
# data.sort(key=lambda tup:tup[1], reverse=True)
book_list.sort(key=lambda dic:dic['comment'], reverse=True)
print(book_list)
# for book in book_list:
#     print(book)

# 数据存储
import pandas as pd
df=pd.DataFrame(book_list)
df.to_csv("DangDang_statistics.csv", index=False)

# 数据可视化
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.figure(figsize=(15,8))
book_name = [v['name'][:15] for v in book_list[:10]]
y = [v['comment'] for v in book_list[:10]]
x = range(len(book_name))
plt.xticks(x, book_name, rotation=30)
plt.bar(x, y)
plt.show()


