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


def create_pipelines(
    document_store,
    preprocessor,
    converter,
    retriever,
    reader,
    translator = None
    ):
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, inputs=["Query"], name="Retriever")
    query_pipeline.add_node(component=reader, inputs=["Retriever"], name="Reader")
    if translator:
        query_pipeline.add_node(component=translator, inputs=["Reader"], name= "Translator")
    index_pipeline = Pipeline()
    index_pipeline.add_node(component=converter, inputs=["File"], name="TextConverter")
    index_pipeline.add_node(component=preprocessor, inputs=["TextConverter"], name="Preprocessor")
    index_pipeline.add_node(component=retriever, inputs=["Preprocessor"], name="Retriever")
    index_pipeline.add_node(component=document_store, inputs=["Retriever"], name="DocumentStore")
    return query_pipeline, index_pipeline


# RETRIEVERS
# 1. DPR retriever
# 2. Embedding retriever
# 3. BM24 retriever

# READERS
# 1. deepset/xlm-roberta-large-squad2 
# 2. deepset/roberta-base-squad2-covid + Translator
class 
xlm_sparse_pipe, sparse_idx_pipe = create_pipelines(
    document_store= ElasticsearchDocumentStore(index="sparse_index", recreate_index=True),
    preprocessor = PreProcessor (language='el', split_by='word', split_length=512,  split_respect_sentence_boundary=True, progress_bar=True),
    converter= TextConverter(valid_languages=['el']),
    retriever = BM25Retriever(),
    reader= FARMReader('xlm-roberta-large-squad2', use_gpu=False)
)

xlm_dense_pipe, dense_idx_pipe = create_pipelines(
    document_store= ElasticsearchDocumentStore(index="sparse_index", recreate_index=True),
    preprocessor = PreProcessor (language='el', split_by='word', split_length=512,  split_respect_sentence_boundary=True, progress_bar=True),
    converter= TextConverter(valid_languages=['el']),
    retriever = EmbeddingRetriever( document_store=document_store,
                                embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                model_format="sentence_transformers" ),
    reader= FARMReader('xlm-roberta-large-squad2', use_gpu=False)
)





retriever = EmbeddingRetriever( document_store=,
                                embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                model_format="sentence_transformers" ),
reader  = FARMReader('xlm-roberta-large-squad2', use_gpu=False)
)





        #retriever = EmbeddingRetriever( document_store=document_store,
             #               embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
              #              model_format="sentence_transformers" )
    #document_store.update_embeddings(retriever)
# index documents.
#index_pipeline.run(file_paths=['data/who/answers.el.txt'])
