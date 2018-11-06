from selenium import webdriver
from htmldom import htmldom
import nltk.data
from nltk.tag import pos_tag 
from nltk.tokenize import word_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#browser= webdriver.Chrome('driver/chromedriver')
browser = webdriver.Firefox()
browser.get('http://text-processing.com/demo/sentiment/')
fp = open("test_reviews.txt")
data = fp.read()
d = []
d =tokenizer.tokenize(data)
print(d)
for i in range(len(d)):
    print(d[i])
    try:
        elem = browser.find_element_by_tag_name('textarea')
    except:
        print('Was not able to find an element with that name.')
    elem.clear()
    elem.send_keys(d[i])
    elem.submit()
    text=browser.page_source
    dom = htmldom.HtmlDom()
    dom = dom.createDom(text)
    p = dom.find( "strong" ).text()
    l=len(p)
    n=str(p)
    #print(l)
    #print(n)
    s=(n[120:l-96])
    print(s)
