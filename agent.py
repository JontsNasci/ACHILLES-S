#pylint:disable=E0001
import requests
import bs4

# Define the URL of the arXiv search results page
url = "https://arxiv.org/search/?query=Material+&searchtype=all&source=header"

urls = []
names = []
full = []

def clean_trasnformer(text):
	return text
	
def fetch(url):
    global urls
    global names
    global full
    # Make a request to the arXiv search results page
    response = requests.get(url)
    
    # Parse the HTML response into a BeautifulSoup object
    soup = bs4.BeautifulSoup(response.content, "html")
    
    # Find all of the article results
    articles = soup.find_all("li", class_="arxiv-result")
    
    # Loop over all of the article result

    
    #print("xx", articles)
    for article in articles:
    
        # Get the article title
        title = article.find("p", class_="title").text
        
        names.append(title)
    
        # Get the article abstract
        abstract = article.find("span", class_="abstract-full").text
        
        full.append(abstract)
    
        # Get the article PDF link
        pdf_link = article.find("span").find("a").get("href")
        
        urls.append(pdf_link)
        # Print the article title, abstract, and PDF link
        
      #  print(title, abstract, pdf_link)
    
#    print(urls)


import requests
import os

words = "abcdefghijklmnopqrstuvwxyz0987654321"
words = list(words)

def test(k):
	r = True
	global words
	for w in k:
		if w.lower() not in words:
			r = False
	return r

def clrStr(s):
	o = ''
	last = ''
	for k in s.split(' '):
		if test(k) and last is not k:
			o += k + '.'
		last = k
	return o.replace('..','').replace('...','').replace('.',' ')

for n in range(len(names)):
	names[n] = clrStr(names[n])

def g(id):
	global names
	global urls
	# The URL of the file to download
	url = urls[id]
	
	# The path to the directory to download the file to

	import os
	
	names[id] = names[id].replace("        ", "")
	names[id] = names[id].replace("      ", "")
	names[id] = names[id].replace("    ", "")
	names[id] = names[id].replace('\n',"")
	names[id] = names[id].replace('/', "")
	names[id] = names[id].replace('\\', '')
	
	print(names[id], 'xxx')
	path = '/storage/emulated/0/direct/' + names[id] +'/'
	directory = path
	
	if not os.path.exists(directory):
	    os.makedirs(directory, exist_ok=True)
	
	# Download the file
	response = requests.get(url)
	with open(os.path.join(path, 'main.pdf'), 'wb') as f:
	    f.write(response.content)
	import PyPDF2

	pdf_reader = PyPDF2.PdfReader(open(path + 'main.pdf', 'rb'))
	fw = open(os.path.join(path, 'main.txt'), 'wb')

	for page in pdf_reader.pages:
		fw.write(clean_transformer(page.extract_text()).encode())
	fw.close()
	
	fw = open(os.path.join(path, "abstract.txt"), 'wb')
	fw.write(clean_transformer(full[id]).encode())
	fw.close()
	print(url, names[id])

def getall(url):
    global urls
    urls = []
    fetch(url)
    for n in range(len(urls)):
    	g(n)

URL = "https://arxiv.org/search/?query=Material+&searchtype=all&source=header"

def v(c):
    r = ""
    for j in c:
        r += j
    return r

import itertools as i

c = 0

for k in range(4):
    combs = i.product("abcdefghijklmnopqrstuvwxyz", repeat=k)
    for n in combs:
        c += 1
        print(c)
        if c >= 5:
           
             print("            ", v(n), "            yyyy")
          
             getall("https://arxiv.org/search/?query=" +v(n)+ "+&searchtype=all&source=header")



