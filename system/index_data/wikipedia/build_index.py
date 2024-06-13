#!/usr/bin/env python3
# Apply  pre-processing and domain filtering (optional) on fetched wikipedia dump articles before saving to a document store index.  
# input file: each line should be a wikipedia articles in json format

import logging
import tempfile
import argparse
import json
import os
from pathlib import Path

from haystack.pipelines import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, JsonConverter

from preprocess import preprocess_wiki
from filter_index import KeywordFilterer

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)

PREPROCESS_FN = None
DOMAIN_FILTERING = False

def init (preprocess=False, apply_filter=False):
    global PREPROCESS_FN
    global DOMAIN_FILTERING
    if preprocess:
        PREPROCESS_FN = preprocess_wiki
    if apply_filter:
        DOMAIN_FILTERING = True

def load_docs (filename):
    with open (filename, "r") as fp:
        return [json.loads(line) for line in fp.readlines()]

def run_index_pipeline(file_paths, index):
    """haystack pipeline for indexing wikipedia data to document store. If --apply_filter, an extra node for keeping only covid-19 related article is added to the pipeline"""
    
    global FILTERING
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        split_by = "word",
        split_length=256,
        split_respect_sentence_boundary=True,
        language= 'el'
        )
    
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=index, recreate_index=True)
 
    p = Pipeline()
    p.add_node(component=JsonConverter(), name = "JsonConverter", inputs=["File"])
    p.add_node(component=preprocessor, name="Preprocessor", inputs=["JsonConverter"])

    previous_node = ["Preprocessor"]

    if DOMAIN_FILTERING:
        p.add_node(component=KeywordFilterer(), name= "KeywordFilterer", inputs =["Preprocessor"])
        previous_node = ["KeywordFilterer"]

    p.add_node(component=document_store, name="DocumentStore", inputs=previous_node)


    p.run_batch(file_paths=file_paths)


def main (data_filename, index):
    global PREPROCESS_FN
    """Index wiki documents in ES doument store

    Args:

    """
    if not os.path.isfile(data_filename):
        raise RuntimeError('%s This is not a path to a filename' % data_filename)
    
    if PREPROCESS_FN is not None:
        documents = PREPROCESS_FN(data_filename)
    else:
        documents = load_docs(data_filename)
    
    temp_dir = tempfile.TemporaryDirectory()

    file_paths = []
    for doc in documents:
        file_name = doc["id"] + ".json"
        file_path = Path(temp_dir.name) / file_name
        file_paths.append(str(file_path))
        with open(file_path, "w") as f:
            f.write(json.dumps(doc, ensure_ascii=False))

    run_index_pipeline(file_paths, index)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename', type=str, help='/path/to/data_filename')
    parser.add_argument('--index_name', type=str, default='test', help='name of index in document store. If already exists, it will be overwritted')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess wiki documents')
    parser.add_argument('--apply_filter', action='store_true',
                        help='filter wiki documents by applying keyword matching')
    args = parser.parse_args()

    init(preprocess=args.preprocess, apply_filter=args.apply_filter)

    main(
        args.data_filename,
        args.index_name
    )

