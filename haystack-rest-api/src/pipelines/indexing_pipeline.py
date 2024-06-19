

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, JsonConverter, TextConverter
from haystack import Pipeline
import os 
host = os.getenv("DOCUMENTSTORE_PARAMS_HOST", "localhost")
port = os.getenv("DOCUMENTSTORE_PARAMS_PORT", 9200)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    split_by = "word",
    split_length=256,
    split_respect_sentence_boundary=True,
    language= 'el'
    ) 

document_store = ElasticsearchDocumentStore(host=host, port=host, username="", password="", index="test_api", recreate_index=True, duplicate_documents="overwrite")

indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=TextConverter(), name = "FileConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["JsonConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Preprocessor"])
