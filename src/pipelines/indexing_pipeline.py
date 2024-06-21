

from haystack.nodes import PreProcessor, TextConverter
from haystack import Pipeline
from pipelines.document_store import document_store

DS = document_store

def indexing_pipeline (recreate_index:str = False):

    global DS
    
    DS.recreate_index = recreate_index

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
    indexing_pipeline.add_node(component=DS, name="DocumentStore", inputs=["Preprocessor"])

    return indexing_pipeline
