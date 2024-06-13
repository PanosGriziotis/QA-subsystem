from haystack.nodes import Crawler
from haystack.pipelines import Pipeline
from haystack.nodes import Crawler, PreProcessor , PDFToTextConverter, DocxToTextConverter
import json
from haystack.document_stores import ElasticsearchDocumentStore
import os
import spacy 
from typing import List
import logging
from tqdm import tqdm
import unicodedata
import glob
import shutil
import re
import nltk
import argparse

from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)

eody_crawler = Crawler(urls = ["https://eody.gov.gr/neos-koronaios-covid-19/", "https://eody.gov.gr/disease/"], crawler_depth=2, filter_urls = ["https://eody.gov.gr/"], output_dir='crawled_files')
eody_crawler_1
moh_crawler = Crawler(urls = ["https://www.moh.gov.gr/articles/health/"],  crawler_depth=1, filter_urls = ["https://www.moh.gov.gr/articles/health/"], output_dir='crawled_files' )

INCLUSION_KEYWORDS = ["sars-cov-2", "covid-19", "covid", "covid19" "koronoios", "koronoioy", "koronaios", "koronaioy",  "pandhmias", "pandhmia"]
EXCLUSION_KEYWORDS = ["epidimiologikis-epitirisis", "weekly-report"]

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_header_footer=True,
    clean_whitespace=True,
    split_by="word",
    split_length=256,
    split_respect_sentence_boundary=True,
    language='el'
)
pdf_converter = PDFToTextConverter (
    remove_numeric_tables= False,
    valid_languages = ['el'],
    ocr = None,
    multiprocessing = True)
doc_converter =  DocxToTextConverter(remove_numeric_tables=False, valid_languages=["el"])

def clean_text(text):

    # normalize unicode escape sequences
    text = unicodedata.normalize('NFC', text)
    # Remove CSS styles within { } braces and any trailing text after the closing brace
    text = re.sub(r'\{[^{}]*\}.*?', '', text, flags=re.DOTALL)
    
    # Remove CSS styles within <style> tags
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    
    # Remove remaining CSS classes and media queries
    text = re.sub(r'(@media.*?\{.*?\})', '', text, flags=re.DOTALL)
    text = re.sub(r'\.[\w-]+\s*\{[^{}]*\}', '', text, flags=re.DOTALL)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove any remaining CSS class names
    text = re.sub(r'\.[\w-]+', '', text)
    
    # Remove multiple consecutive whitespace characters
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text

def check_for_keywords(text):
    # Check inclusion keywords
    text = text.lower()
    for keyword in INCLUSION_KEYWORDS:
        if re.search(keyword.lower(), text):
            # Check exclusion keywords
            for ex_keyword in EXCLUSION_KEYWORDS:
                if re.search(ex_keyword.lower(), text):
                    return False
            return True
    
    return False

def get_url_base(url):
    """Extract keywords from a URL"""
    # Parse the URL
    parsed_url = urlparse(url)
    # Get the path component from the URL
    path = parsed_url.path
    # Split the path into parts using '/' as delimiter
    path_parts = path.split('/')

    # Extract the filename from the last part of the path
    url_base = path_parts[-1]


    return url_base

def filter_out_url(url):
    url_base = get_url_base(url)
    return not check_for_keywords(url_base) # not False = True = remove url / not True = False = keep url

def filter_out_file(filename):
    """Filter out pdf files based on keywords in their basename"""
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    return not check_for_keywords(basename) # not False = True = remove file / not True = False = keep file

def process_json_files(json_files: list):
    """process json files with crawled html content"""

    docs = []
    logging.info(f"Processing crawled html files...")

    for json_file in tqdm(json_files):

        with open(json_file, "r") as fp:

            doc = json.load(fp)
            
        
            url = doc["meta"]["url"]

            if filter_out_url(url):
                continue

            text = clean_text(doc["content"])

            if text:
                docs.append({"content": text, 'url': url})

    return docs

def process_files(files: list):
    """process other files downloaded while parsing html pages (only pdf and docx)"""

    allowed_extensions = [".pdf", ".docx"]

    extracted_docs = []
    logging.info(f"Processing PDF files...")

    for file in tqdm(files):

        extension = get_file_suffix(file)
        
        if filter_out_file(file) and extension not in allowed_extensions:

            os.remove(file)
            
            continue

        if extension == ".pdf":
            doc = pdf_converter.convert(file)[0]
        elif extension == ".docx":
            doc = doc_converter.convert(file)[0]
        
        doc.content = clean_text(doc.content)
        extracted_docs.append(doc)

        os.remove(file)

    return extracted_docs

def get_file_suffix(file_path):
    # Extract the file suffix using os.path.splitext
    _, file_suffix = os.path.splitext(file_path)
    return file_suffix

def preprocess():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    crawled_files_dir = os.path.join(script_dir, "crawled_files")

    # Get all JSON file paths in the "crawled_files" subdirectory
    json_files = glob.glob(os.path.join(crawled_files_dir, "*.json"))

    # Process JSON files and delete the "crawled_files" subdirectory
    docs_from_json = process_json_files(json_files)
    shutil.rmtree(crawled_files_dir)

    script_name = os.path.basename(__file__)
    
    # Iterate through all files in the directory
    crawled_files = []
    for file_name in os.listdir(script_dir):
        file_path = os.path.join(script_dir, file_name)
        
        #    Skip directories and the script itself
        if os.path.isdir(file_path) or file_name == script_name:
            continue
        crawled_files.append(file_path)

    # Extract text from remaining files in directory (only PDF/DOCX)
    docs = process_files(files=crawled_files)

    # Merge the two lists of documents
    all_docs = docs_from_json + docs

    # preprocess doc (add split, clean, remove footer, convert to haystack Document format)
    return preprocessor.process(all_docs)

def crawl (crawler):
    crawler.crawl()
    docs = preprocess()
    return docs

def run_web_crawler():
    crawled_docs = []

    for crawler in [eody_crawler, moh_crawler]:
        docs = crawl(crawler=crawler)
        crawled_docs.append(docs)
    
    return crawled_docs      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='filename to save crawled documents')
    args = parser.parse_args()

    crawled_docs = run_web_crawler()

    with open (os.path.join(args.save_dir, 'crawled_doc.json'), "w") as fp:
        for doc in crawled_docs:
            doc = json.dumps(doc[0].to_dict(), ensure_ascii=False)
            fp.write(doc + "\n")
    

