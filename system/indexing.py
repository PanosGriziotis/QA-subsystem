from pathlib import Path
from typing import List
from haystack.nodes import TextConverter
from haystack.nodes import PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore

def convert_to_haystack_format(file):
    
    # convert txt files to dicts
    converter = TextConverter(valid_languages=["el"])
    docs = converter.convert(file_path=file, meta=None)
    # clean and split
    preprocessor = PreProcessor(
        language='el',
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=180,
        split_respect_sentence_boundary=True,
    )
    cleaned_docs = preprocessor.process(docs)
    return cleaned_docs

def index_document_store(docs):
    document_store = ElasticsearchDocumentStore()
    try:
      document_store.write_documents(docs)
      return document_store
    except Exception as e:
      print(e)


if __name__ == '__main__':
    docs = convert_to_haystack_format('../data/covid_data.txt')
    index_document_store(docs)