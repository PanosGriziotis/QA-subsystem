
from typing import List, Dict, Any, Optional, Union
from haystack.nodes import FARMReader, EmbeddingRetriever, SentenceTransformersRanker
from haystack.pipelines import Pipeline
import os 
import sys 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from document_store.initialize_document_store import document_store as DOCUMENT_STORE

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")
    
retriever = EmbeddingRetriever(embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2", document_store=DOCUMENT_STORE)
ranker = SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco")
reader = FARMReader(model_name_or_path="panosgriz/mdeberta-v3-base-squad2-covid-el_small")

p = Pipeline()
p.add_node(component=retriever, name ="Retriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
p.add_node(component=reader, name="Reader", inputs=["Ranker"])

extractive_qa_pipeline = p