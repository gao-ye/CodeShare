# -*- coding:UTF-8 -*-
import requests
from bs4 import BeautifulSoup
if __name__ == '__main__':
    target = 'http://www.biqukan.com/1_1094/'
    req = requests.get(url=target)
    req.encoding = req.apparent_encoding
    html = req.text
    div_bf = BeautifulSoup(html)
    div = div_bf.find_all('div', class_ = 'listmain')  ##返回的只有一个元素的列表
    # print(div)
    # print(div[0])
    a_bf =BeautifulSoup(str(div[0]))
    for each in a_bf:
        print(each.string)