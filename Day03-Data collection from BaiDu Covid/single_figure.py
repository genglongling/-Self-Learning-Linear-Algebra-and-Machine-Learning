url = 'http://img3m0.ddimg.cn/47/6/28486010-1_b_58.jpg'
headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
import requests
from matplotlib import pyplot as plt

response = requests.get(url, headers=headers)
if response.status_code == 200:
    img_Data = response.content #二进制
    with open('./images/image1.jpg', mode='wb') as f:
        f.write(img_Data)
    #plt.imshow(img_Data)


