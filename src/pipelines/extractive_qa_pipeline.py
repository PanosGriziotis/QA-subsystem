
from typing import List, Dict, Any, Optional, Union
from haystack.nodes import FARMReader, EmbeddingRetriever
from haystack.pipelines import Pipeline
import os 
import sys
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from document_store.initialize_document_store import document_store as DOCUMENT_STORE
from pipelines.ranker import SentenceTransformersRanker

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")
    
retriever = EmbeddingRetriever(
    embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2",
    document_store=DOCUMENT_STORE,
    top_k= 10
    )
ranker = SentenceTransformersRanker(
    model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco",
    scale_score=True,
    top_k=10
    )
reader = FARMReader(
    model_name_or_path="panosgriz/mdeberta-v3-base-squad2-covid-el_small",
    use_gpu=True,
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    use_confidence_scores=True,
    top_k = 3
    )

p = Pipeline()
p.add_node(component=retriever, name ="Retriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
p.add_node(component=reader, name="Reader", inputs=["Ranker"])

extractive_qa_pipeline = p