import requests
from bs4 import BeautifulSoup as bs
#from  PyPDF2  import PdfFileReader
import os
import sys
import spacy 

def get_true_case (self, text):

    "Apply a truecaser to text based on name entities"
    
    spacy_nlp = spacy.load("el_core_news_sm")
    words = [word.text for word in spacy_nlp(text)]
    ents = [ent.text for ent in text.ents]
    capitalized_words = [w.capitalize() if w in ents else w for w in words]
    true_cased_words = [w.lower() if w.isupper() else w for w in capitalized_words]

    return ' '.join(true_cased_words)

def download_texts(main_url, data_dir):

    """input: main url path
       output: txt files containing scraped text from all the embedded urls"""
    
    # get a list of all embedded urls
    urls = [] 
    response = requests.get (main_url)
    soup = bs (response.content, 'lxml')
    container = soup.find(id="main-container")
    for a_href in container.find_all("a",href=True):
        url = a_href["href"]
        if "https://eody.gov.gr" in url and url not in urls:
            urls.append(url)
    
    if  os.path.isdir(data_dir) == False:
        os.mkdir(data_dir)
    os.chdir(data_dir)

    i = 0
    for url in urls[7:]: # getting rid of urls relevant to daily reports
        response = requests.get(url)
        content_type = response.headers.get('content-type')
        if content_type == 'text/html; charset=UTF-8':
            i+=1
            soup = bs(response.content, 'lxml')
            ps = soup.find_all('p')   
            passage = [el.get_text() for el in ps[:-5]] # [:-5] to get rid of last paragraphs related to acessibility protocols 
            passage = ' '.join(passage)
            if passage not in ['\xa0','',' ']:
                with open (f'doc{i}.txt', 'w', encoding='utf8') as doc_file:
                    doc_file.write(get_true_case(passage))


if __name__ == '__main__': 
    URL = "https://eody.gov.gr/neos-koronaios-covid-19/"
    download_texts(URL,sys.argv[1])
    


