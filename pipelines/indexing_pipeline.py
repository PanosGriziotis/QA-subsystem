

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import logging

from haystack.nodes import EmbeddingRetriever, FileTypeClassifier, JsonConverter, TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack import Pipeline

from utils.file_type_classifier import JsonFileDetector
from document_store.initialize_document_store import document_store as DOCUMENT_STORE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")

json_converter =JsonConverter(valid_languages=["el"])
file_type_classifier = FileTypeClassifier()
text_converter = TextConverter(valid_languages=['el'])
pdf_converter = PDFToTextConverter(valid_languages=['el'])
docx_converter = DocxToTextConverter(valid_languages=['el'])
retriever = EmbeddingRetriever(embedding_model="panosgriz/covid_el_embedding_retriever", document_store=DOCUMENT_STORE, max_seq_len=128)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    split_by = "word",
    split_length=128,
    split_respect_sentence_boundary=True,
    language= 'el'
    )

DOCUMENT_STORE.recreate_index = True

p = Pipeline()

# Classify doc according to extension and routes it to corresponding converter
p.add_node (component=JsonFileDetector(), name="JsonFileDetector", inputs=["File"])
p.add_node(component=json_converter, name="JsonConverter", inputs=["JsonFileDetector.output_1"])
p.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["JsonFileDetector.output_2"])
p.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
p.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
p.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
# Split, clean and convert document(s) to haystack Document object(s)
p.add_node(component=preprocessor, name="Preprocessor", inputs=["JsonConverter", "TextConverter", "PdfConverter", "DocxConverter"])
# Update the document embeddings in the the document store using the encoding model specified in the retriever
p.add_node(component=retriever, name = "DenseRetriever", inputs=["Preprocessor"])
p.add_node(component=DOCUMENT_STORE, name= "DocumentStore", inputs=["DenseRetriever"])
indexing_pipeline = p
