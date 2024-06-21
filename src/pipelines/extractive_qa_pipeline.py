from typing import List, Dict, Any, Optional, Union
from pipelines.document_store import document_store
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import Pipeline
import os

# TODO add log for checking if index has documents

DS = document_store

def extractive_qa_pipeline():
    """"""
    # TODO set default parameters

    global DS

    retriever = BM25Retriever(document_store=DS, top_k=1)
    reader = FARMReader(model_name_or_path="panosgriz/mdeberta-v3-base-squad2-covid-el_small", top_k=1)

    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name ="Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

    return pipeline

if __name__ == "__main__":
    # ============TEST PIPELINE================
    result=extractive_qa_pipeline().run(query="Πώς μεταδίδεται ο covid;")
    print (result["answers"][0].answer)
