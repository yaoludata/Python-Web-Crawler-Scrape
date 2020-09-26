# Web crawler: get movie TOP250 information
# Problem requirement:
# crawl all the serial numbers/movie names/ratings/recommendations/links in Movie TOP250, and the result is to display and print them all

# import requests and BeautifulSoup
import requests
from bs4 import BeautifulSoup

# Crawl down the serial number/film name/rating/recommendation/link in the movie TOP250
for x in range(10): # 10 pages
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'} #deal with the 418 code status problem
    url = 'https://movie.douban.com/top250?start={}&filter='.format(x * 25) # movie TOP250
    res = requests.get(url, headers=headers)
    html = res.text
    soup = BeautifulSoup(html,'html.parser')
    items = soup.find('ol', class_='grid_view')
    for titles in items.find_all('div', class_='item'):
        num = titles.find('em', class_="") # serial number
        title = titles.find('span', class_='title') # film name
        rating = titles.find('span', class_='rating_num') #rating
        url = titles.find('a')
        if titles.find('span', class_='inq') != None: # some films do not have recommendation
           comment = titles.find('span', class_='inq').text # recommendation
        else:
           comment = ""
        print(num.text + '.' + title.text + ' —— ' + 'Rating:' + rating.text + '\n'  + 'Comment：' + comment +'\n' + url['href']) # link
