from typing import List, Optional
from haystack import Document
from haystack.nodes import TextConverter
from haystack.nodes import PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore, OpenSearchDocumentStore
from haystack.nodes import FARMReader,  BM25Retriever, Seq2SeqGenerator, TransformersTranslator, EmbeddingRetriever, RAGenerator
from haystack.pipelines import ExtractiveQAPipeline, TranslationWrapperPipeline
from haystack.nodes.label_generator import PseudoLabelGenerator
from haystack import Pipeline
from transformers import T5Tokenizer
from haystack.utils import launch_es
from converter import _MT5inputConverter


if __name__ == '__main__':
    
    preprocessor = PreProcessor (language='el', split_by='word', split_length=512,  split_respect_sentence_boundary=True, progress_bar=True)
    document_store = ElasticsearchDocumentStore()
    retriever  =  BM25Retriever(document_store= document_store)
    #document_store = ElasticsearchDocumentStore(embedding_dim = 384, index="documents")
    generator = Seq2SeqGenerator(model_name_or_path='google/mt5-small' ,  input_converter = _MT5inputConverter())
    #reader  = FARMReader('system/last_model', use_gpu=False)

    # index pipeline
    index_pipeline = Pipeline()
    index_pipeline.add_node(component=TextConverter(valid_languages='el'), inputs=["File"], name="TextConverter")
    index_pipeline.add_node(component=preprocessor, inputs=["TextConverter"], name="Preprocessor")
    index_pipeline.add_node(component=document_store, inputs=["Preprocessor"], name="DocumentStore")

    index_pipeline.run(file_paths=['data/who/answers.el.txt'])
   
    # query pipeline 
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, inputs=["Query"], name="Retriever")
    #query_pipeline.add_node(component=reader, inputs=["Retriever"], name= "Reader")
    query_pipeline.add_node(component=generator, inputs=["Retriever"], name= "Generator")
    
    result = query_pipeline.run(query="Τι πρέπει να κάνω αν έχασα την εκπαίδευσή μου λόγω της πανδημίας COVID-19;", params={"Reader" : {"top_k":3} , "Retriever": {"top_k": 3}})
    print (result['answers'])


        #retriever = EmbeddingRetriever( document_store=document_store,
             #               embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
              #              model_format="sentence_transformers" )
    #document_store.update_embeddings(retriever)
