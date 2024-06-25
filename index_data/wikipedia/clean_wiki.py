#!/usr/bin/env python3

import argparse
import json
import os
import logging
import time

from tqdm import tqdm

import regex as re
import unicodedata
import html

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def normalize(text):
    """Normalize text and remove disambiguation, list, category and prototype wiki pages."""
    return unicodedata.normalize('NFC', text.replace('\n', ' '))

def remove_structured_pages(article):
    """Remove articles from special categories"""

    for k, v in article.items():      
        article[k] =  html.unescape(normalize(v))

    if '(αποσαφήνιση)'in article['title'].lower():
        return None

    if re.match(r'(Κατάλογος .+)|(Κατηγορία:.+)|(Πρότυπο:.+)',
                article['title']):
        return None

    return {'id': article['id'], 'title': article['title'], 'content': article['text']}

def preprocess_wiki_docs(filename):
    """Parses the contents of a file with wikipedia data. Each line is a JSON encoded article."""
    
    logging.info(f"Starting to preprocess file: {filename}")
    start_time = time.time()

    documents = []
    with open(filename, "r") as f:
        for line in tqdm(f.readlines()):
            doc = remove_structured_pages(json.loads(line))
            if not doc:
                continue
            documents.append(doc)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Finished preprocessing file: {filename} in {elapsed_time:.2f} seconds")
    logging.info (f"Number of wiki articles after preprocessing: {len(documents)}")

    return documents

if __name__ == '__main__':

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='/path/to/data/filename.txt')
    args = parser.parse_args()

    basename = os.path.splitext(os.path.basename(args.file_path))[0]
    dir = os.path.dirname(args.file_path)
    output_file = output_file = os.path.join(dir, f"{basename}_processed.txt")

    with open(output_file, "w", encoding="utf-8") as writer:
        docs = preprocess_wiki(args.file_path)
        for doc in docs:
            json_string = json.dumps(doc, ensure_ascii=False)
            writer.write(json_string + "\n")

    logging.info(f"Processing complete. Output written to: {output_file}")
    """