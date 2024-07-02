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

# Get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory of SCRIPT_DIR (which should be my_project) to the Python path
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))

from haystack.nodes import Crawler
from utils.file_type_classifier import init_file_to_doc_pipeline
from utils.data_handling_utils import is_english

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

def get_file_suffix (filename):
    return os.path.splitext(filename)[1]

def get_file_basename (filename):
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    return basename

def filter_out_file(basename:str):
    """Discard files according to keyword filtering"""
    return not check_for_keywords(basename) 

def process_files(files: list):
    """
    Process crawled file. Return cleaned and keyword relevant documents only.
    Keyword checking is applied on the files' basename level.
    """

    processed_docs = []
    logging.info(f"Processing downloaded files...")
    
    file_basenames = []
    for file in tqdm(files):

        file_basename = get_file_basename(file)
        if get_file_suffix(file) not in ['.jsonl', '.json', '.txt', '.pdf', '.docx']:
            continue

        # check for inclusion & exclusion keywords on file's basename
        if filter_out_file(file_basename):
            os.remove(file)
            continue

        if file_basename in file_basenames:
            os.remove(file)
            continue
        # Run file classifier, extract text and convert to haystack Document object        
        file_converter_pipeline = init_file_to_doc_pipeline()
        docs = file_converter_pipeline.run(file_paths=[file])["documents"]
        
        for doc in docs:
            # check for remaining english content
            if is_english(doc.content):
                continue
            # post process documents
            doc.content = clean_text(doc.content)

            doc.meta = {'filename': file_basename}
            processed_docs.append(doc)
        
        file_basenames.append(file_basename)
        os.remove(file)

    return processed_docs

def convert_crawled_data_to_docs():
    """convert all fetched data to haystack Document objects"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    
    # Iterate through all files in running script directory
    crawled_files = []
    for file_name in os.listdir(script_dir):
        file_path = os.path.join(script_dir, file_name)
        #  Skip sub-directories and the script itself
        if os.path.isdir(file_path) or file_name == script_name:
            continue
        crawled_files.append(file_path)

    docs = process_files(files=crawled_files)
    
    return docs

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
        extract_hidden_text=True,
        filter_urls = [
            "https://eody.gov.gr/"
        ],
        output_dir="./")
    
    moh_crawler = Crawler(urls = [
        "https://www.moh.gov.gr/articles/health/"
        ],
        crawler_depth=2,
        extract_hidden_text=True,
        filter_urls = [
            "https://www.moh.gov.gr/articles/health/"
        ],
        output_dir="./")

    crawled_docs = []

    for crawler in [eody_crawler, moh_crawler]:
        docs = run_web_crawler(crawler)
        crawled_docs.append(docs)
    
    # save all processed crawled docs in a .json file
    file_path = os.path.join(args.save_dir, 'crawled_docs.jsonl')
    print(f"Saving crawled docs to {file_path}")

    crawled_docs = [item for sublist in crawled_docs for item in sublist]

    with open(file_path, "w") as fp:
        for doc in tqdm(crawled_docs):
            if type(doc) is not dict:
                doc = doc.to_dict()
            doc_json = json.dumps(doc, ensure_ascii=False)
            fp.write(doc_json + "\n")