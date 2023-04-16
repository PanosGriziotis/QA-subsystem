from typing import List, Optional
from haystack import Document
from haystack.nodes import TextConverter
from haystack.nodes import PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore, OpenSearchDocumentStore
from haystack.nodes import FARMReader,  BM25Retriever,DensePassageRetriever, TransformersTranslator, EmbeddingRetriever, RAGenerator
from haystack.pipelines import ExtractiveQAPipeline, TranslationWrapperPipeline
from haystack.nodes.label_generator import PseudoLabelGenerator
from haystack import Pipeline
from transformers import T5Tokenizer
from haystack.utils import launch_es
from typing import List

class PipelineCreator ():
    def __init__(self) -> None:
        self.document_store = self.document_store   

    def create_pipelines(self, 
        preprocessor,
        converter,
        retriever,
        reader):

        query_pipeline = Pipeline()
        query_pipeline.add_node(component=retriever, inputs=["Query"], name="Retriever")
        query_pipeline.add_node(component=reader, inputs=["Retriever"], name="Reader")

        index_pipeline = Pipeline()
        index_pipeline.add_node(component=converter, inputs=["File"], name="TextConverter")
        index_pipeline.add_node(component=preprocessor, inputs=["TextConverter"], name="Preprocessor")
        index_pipeline.add_node(component=retriever, inputs=["Preprocessor"], name="Retriever")
        index_pipeline.add_node(component=self.document_store, inputs=["Retriever"], name="DocumentStore")
        
        return query_pipeline, index_pipeline

# RETRIEVERS
# 1. DPR retriever
# 2. Embedding retriever
# 3. BM24 retriever

# READERS
# 1. fine_tuned_reader
