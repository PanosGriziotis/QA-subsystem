from typing import List, Dict, Any, Optional, Union
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import Pipeline
import os
import logging

from document_store.initialize_document_store import document_store as DOCUMENT_STORE


if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")
    

# TODO add log for checking if index has documents
def extractive_qa_pipeline():
    """"""
    # TODO set default parameters

    global DOCUMENT_STORE 

    retriever = BM25Retriever(document_store=DOCUMENT_STORE, top_k=1)
    reader = FARMReader(model_name_or_path="panosgriz/mdeberta-v3-base-squad2-covid-el_small", top_k=1)

    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name ="Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

    return pipeline

if __name__ == "__main__":
    # ============TEST PIPELINE================
    result=extractive_qa_pipeline().run(query="Πώς μεταδίδεται ο covid;")
    print (result["answers"][0].answer)
