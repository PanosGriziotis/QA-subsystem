from typing import List
import logging

import json
import os
from tqdm import tqdm
import unicodedata
import glob
import shutil
import re
import argparse
from urllib.parse import urlparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from haystack.nodes import Crawler
from utils.file_type_classifier import classify_and_convert_file_to_docs

logging.basicConfig(level=logging.INFO)


def clean_text(text:str):
    """Clean scraped text"""

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

def check_for_keywords(text:str):
    """Filter text based on given inclusion & exclusion keywords."""
    
    text = text.lower()
    for keyword in INCLUSION_KEYWORDS:
        if re.search(keyword.lower(), text):
            
            for ex_keyword in EXCLUSION_KEYWORDS:
                if re.search(ex_keyword.lower(), text):
                    return False
            return True 
    
    return False 

def get_url_base(url:str):
    """Get base name from a URL."""

    # Parse the URL
    parsed_url = urlparse(url)
    # Get the path component from the URL
    path = parsed_url.path
    # Split the path into parts using '/' as delimiter
    path_parts = path.split('/')
    # Extract the filename from the last part of the path
    url_base = path_parts[-1]

    return url_base

def filter_out_url(url:str):
    """Remove urls according to keyword filtering"""

    url_base = get_url_base(url)
    return not check_for_keywords(url_base) # not False = True = remove url / not True = False = keep url

def get_file_suffix (filename):
    return os.path.splitext(filename)[1]

def get_file_basename (filename):
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    return basename

def filter_out_file(basename:str):
    """Remove downloaded files according to keyword filtering"""
    return not check_for_keywords(basename) 

def process_json_files(json_files: List[str]):
    """
    Process downloaded json files containing crawled html content. 
    Return cleaned and keyword relevant documents only.
    Keyword matching is applied in the basename of the URL
    """

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
    """
    Process other files downloaded while parsing html pages (.pdf, .docx, .txt).
    Return cleaned and keyword relevant documents only.
    Keyword checking is applied on the files' basenames.
    """

    processed_docs = []
    logging.info(f"Processing downloaded files...")

    for file in tqdm(files):

        file_basename = get_file_basename(file)
        if get_file_suffix(file) not in ['.txt', '.pdf', '.docx']:
            continue

        if filter_out_file(file_basename):
            os.remove(file)
            continue

        # Run file classifier, extract text and convert to haystack Document object
        docs = classify_and_convert_file_to_docs(file)['documents']
        for doc in docs:

            doc.content = clean_text(doc.content)
            doc.meta = {'filename': file_basename}
            processed_docs.append(doc)
        
        os.remove(file)

    return processed_docs

def convert_crawled_data_to_docs():
    """convert all fetched data to haystack Document objects"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    crawled_files_dir = os.path.join(script_dir, "crawled_files")

    # Process all JSON files and then delete the "crawled_files" subdirectory
    json_files = glob.glob(os.path.join(crawled_files_dir, "*.json"))
    docs_from_json = process_json_files(json_files)
    shutil.rmtree(crawled_files_dir)
    
    # Iterate through all files in running script directory
    crawled_files = []
    for file_name in os.listdir(script_dir):
        file_path = os.path.join(script_dir, file_name)
        #  Skip sub-directories and the script itself
        if os.path.isdir(file_path) or file_name == script_name:
            continue
        crawled_files.append(file_path)

    # Extract text from remaining files in directory
    docs = process_files(files=crawled_files)

    # Merge the two lists of documents
    all_docs = docs_from_json + docs
    
    return all_docs

def run_web_crawler(crawler):
    """run crawler"""
    crawler.crawl()
    docs = convert_crawled_data_to_docs()
    return  docs  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), help='filename to save crawled documents')
    args = parser.parse_args()

    INCLUSION_KEYWORDS = ["sars-cov-2", "covid-19", "covid", "covid19" "koronoios", "koronoioy", "koronaios", "koronaioy",  "pandhmias", "pandhmia"]
    EXCLUSION_KEYWORDS = ["epidimiologikis-epitirisis", "weekly-report"]

    eody_crawler = Crawler(urls = [
        "https://eody.gov.gr/neos-koronaios-covid-19/",
        ],
        crawler_depth=2,
        filter_urls = [
            "https://eody.gov.gr/"
        ],
        output_dir='crawled_files')
    
    moh_crawler = Crawler(urls = [
        "https://www.moh.gov.gr/articles/health/"
        ],
        crawler_depth=1,
        filter_urls = [
            "https://www.moh.gov.gr/articles/health/"
        ],
        output_dir='crawled_files')

    crawled_docs = []

    for crawler in [eody_crawler, moh_crawler]:
        docs = run_web_crawler(crawler)
        crawled_docs.append(docs)

    file_path = os.path.join(args.save_dir, 'crawled_docs.json')
    print(f"Saving to {file_path}")

    crawled_docs = [item for sublist in crawled_docs for item in sublist]

    with open(file_path, "w") as fp:
        if not crawled_docs:
            print("Warning: crawled_docs is empty.")
        
        for doc in tqdm(crawled_docs):
            if type(doc) is not dict:
                doc = doc.to_dict()
            doc_json = json.dumps(doc, ensure_ascii=False)
            fp.write(doc_json + "\n")