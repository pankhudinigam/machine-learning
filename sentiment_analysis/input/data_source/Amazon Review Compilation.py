from os import listdir
from os.path import isfile,join

#main:
path="F:\\BTP 2015-2016\\AmazonReviews\\mobilephone\\"
fpw=open("Review File.txt","w")
files=listdir(path)
#print(files)
count=0
for file in files:
    fp=open(path+file,"r")
    content=fp.read()
    content_starter=content.find('[')
    content_terminator=content.find('], "ProductInfo":')
    content=content[content_starter+1:content_terminator]
    reviews=content.split('{')
    
    for i in range(len(reviews)):
        temp=reviews[i]
        temp=temp[:temp.find('", "Date": "')]
        temp=temp[temp.find(' "Content": "')+len(' "Content": "'):]
        temp=temp.strip()
        reviews[i]=temp
    
    for review in reviews:
        lst=review.split(".")
    for stmnt in lst:
        stmnt=stmnt.strip()
        if(stmnt is not ""):
            fpw.write(stmnt+".\n")
            count=count+1
print("Done.")
print("count= ",count)
    
