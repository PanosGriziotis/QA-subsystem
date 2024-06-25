from haystack.nodes import PreProcessor, TextConverter
from haystack import Pipeline
from document_store.initialize_document_store import document_store as DOCUMENT_STORE

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")

def indexing_pipeline (recreate_index:str = False):

    global DOCUMENT_STORE
    
    DOCUMENT_STORE.recreate_index = recreate_index

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        split_by = "word",
        split_length=256,
        split_respect_sentence_boundary=True,
        language= 'el'
        )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=TextConverter(), name = "FileConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["FileConverter"])
    indexing_pipeline.add_node(component=DOCUMENT_STORE, name="DocumentStore", inputs=["Preprocessor"])

    return indexing_pipeline
