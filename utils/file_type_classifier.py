# File classifier for: .txt, .pdf, .docx files
from typing import Dict, List, Union
from haystack.schema import Document
from haystack.nodes import FileTypeClassifier, JsonConverter, TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.pipelines import Pipeline
import sys
import os
from haystack.nodes.base import BaseComponent
from tqdm import tqdm 
import logging
from pathlib import Path

file_type_classifier = FileTypeClassifier()
text_converter = TextConverter(valid_languages=['el'])
pdf_converter = PDFToTextConverter(valid_languages=['el'])
docx_converter = DocxToTextConverter(valid_languages=['el'])
json_converter =JsonConverter(valid_languages=["el"])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_header_footer=True,
    clean_whitespace=True,
    split_by="word",
    split_length=256,
    split_respect_sentence_boundary=True,
    language='el'
)

def classify_and_convert_file_to_docs (filepath:str, custom_preprocessor:PreProcessor=None) -> Dict[str, List[Document]]:
    """
    Run file classifier, route file to corresponding converter, and return haystack Document objects extracted from file
    Return: Dict["documents": List[Document]]
    """
    # initialize default preprocessor if not given in function arguments
    global preprocessor
    if custom_preprocessor is not None:
        preprocessor = custom_preprocessor

    p = Pipeline()

    # check for docx, pdf, txt extensions:
    p.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])
    p.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
    p.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
    p.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
    # split and convert to haystack document object
    p.add_node(
        component=preprocessor,
        name="Preprocessor",
        inputs=["TextConverter", "PdfConverter", "DocxConverter"],
    )

    return p.run(file_paths=[filepath])

class JsonFileDetector (BaseComponent):
    """
    Route json or jsonl input file in an Indexing Pipeline to JSONConverter or FileClassifier.
    Note: Only a single file path is allowed as input
    """
    outgoing_edges = 2
    def __init__(self):
        super().__init__()

    def _get_extension(self, file_paths: List[Path]) -> str:
        extension = file_paths[0].suffix.lower()
        return extension
    
    def run(self, file_paths:List[str]):
        
        paths = [Path(path) for path in file_paths]
        extension = self._get_extension(paths)
        output_index = 1 if extension in [".json", ".jsonl"] else 2

        output = {"file_paths": paths}

        return output, f'output_{output_index}'

    def run_batch(
        self,
        **kwargs):
         return

if __name__ == "__main__":

    file = sys.argv[1]
    result = JsonFileDetector().run([file])
    print (result)
    