# File classifier for: .txt, .pdf, .docx files

from haystack.nodes import FileTypeClassifier, TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.pipelines import Pipeline
import sys

file_type_classifier = FileTypeClassifier()
text_converter = TextConverter(valid_languages=['el'])
pdf_converter = PDFToTextConverter(valid_languages=['el'])
docx_converter = DocxToTextConverter(valid_languages=['el'])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_header_footer=True,
    clean_whitespace=True,
    split_by="word",
    split_length=256,
    split_respect_sentence_boundary=True,
    language='el'
)

def classify_and_convert_file(filepath):
    """
    takes a file, classifies it according to its format (docx, pdf, txt)
    and after extracting text, converts it to a haystack Document object
    """

    p = Pipeline()
    p.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])
    p.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
    p.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
    p.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
    p.add_node(
        component=preprocessor,
        name="Preprocessor",
        inputs=["TextConverter", "PdfConverter", "DocxConverter"],
    )

    return p.run(file_paths=[filepath])

if __name__ == "__main__":

    
    file = sys.argv[1]
    doc = classify_and_convert_file(file)

    print (doc)

    