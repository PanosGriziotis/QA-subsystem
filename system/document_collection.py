#from cgitb import htm
#from haystack.nodes.preprocessor import PreProcessor
from haystack.utils.preprocessing import convert_files_to_dicts
import requests
from bs4 import BeautifulSoup as bs
import io
#from  PyPDF2  import PdfFileReader
import os
main_url = "https://eody.gov.gr/neos-koronaios-covid-19/"


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
                    doc_file.write(passage)

        """elif content_type == 'application/pdf':
            pdf= io.BytesIO (response.content)
            reader = PdfFileReader(pdf)
            contents = reader.getPage(0).extractText().split('\n')
            passage = ' '.join (contents) 
            pdf_file.write(passage)"""       
"""  
def clean_docs (data_dir):
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    for txt_file in data_dir:
        preprocessor.process(txt_file)"""  
        
download_texts(main_url,'./data/plain_data')

print ('done')


