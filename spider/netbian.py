#coding=utf-8
import requests
from  bs4  import BeautifulSoup
import re
import os
from tqdm import tqdm
import shutil  

def download(path):
    url = 'http://pic.netbian.com'
    header = {'User-Agent': 'Mozilla/5.0'}

    start_html = requests.get(url, headers=header)
    soup = BeautifulSoup(start_html.text,"html.parser")

    illegal_char=[ ':', '?',  '|']  ##非法字符 图片命名中出现这些字符时无法保存字符 需要替换为其他字符

    #找寻最大页数
    all_a = soup.find('div', class_='page').find_all('a')
    for a in all_a:
        num = a.get_text()
        try:
            num = int(num)
            max_page = num
        except:
            continue
    print("max_page is ", max_page)

    cnt = 0
    # for i in  tqdm(range(100, max_page)):
    for i in  tqdm(range(1, 400)):
    # for i in tqdm(range(1,100)):  ##此处选择下载前100页中的图片
    # for i in range(2, 10):
        if i == 1:
            page_url = 'http://pic.netbian.com'
        else:
            page_url = 'http://pic.netbian.com/index_'+str(i)+'.html'
        
        page_html = requests.get(page_url, headers=header)
        page_html.encoding = page_html.apparent_encoding
        soup = BeautifulSoup(page_html.text,"html.parser")
        
        all_a = soup.find('div', class_ = 'slist').find_all('a', target="_blank")
        for  a in all_a:
            try:
                post_url = a.find('img')['src']
                new_url = url+'/'+post_url
                img = requests.get(new_url, headers=header)
                pic_name = path+a.get_text()+'.jpg'

                for  ch in illegal_char:
                    pic_name = pic_name.replace(':', '')

                f = open(pic_name,'wb')
                f.write(img.content)
                f.close()

            except:
                cnt+=1
                if os.path.exists(pic_name):
                    os.remove(pic_name)

if __name__ == "__main__":
    ##目录处理 将图片保存在 ./pic 目录下
    path = './pic/'
    if os.path.exists(path):
        shutil.rmtree(path)  
    os.mkdir(path)  

    download(path)







