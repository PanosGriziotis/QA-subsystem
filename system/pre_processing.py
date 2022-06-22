from encodings import utf_8
from haystack.nodes import TextConverter
from pathlib import Path
from haystack.nodes import PreProcessor
import spacy
from haystack.utils.preprocessing import convert_files_to_dicts
import json
import os

def truecase_text(text):
    "Applies a truecaser to text based on name entities"
    spacy_nlp = spacy.load("el_core_news_sm")
    text = spacy_nlp(text)
    words = [word.text for word in text]
    ents = [ent.text for ent in text.ents]
    capitalized_words = [w.capitalize() if w in ents else w for w in words]
    true_cased_words = [w.lower() if w.isupper() else w for w in capitalized_words]
    return ' '.join(true_cased_words)


def convert_to_haystack_format(path):
    "splits documents and turns them into dicts"
    all_docs = []
    converter = TextConverter(valid_languages=["el"])
    for file in Path(path).iterdir():
        all_docs.append(converter.convert(file_path=file, meta=None)[0])

    preprocessor = PreProcessor(
        language='el',
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(all_docs)
    return docs

    """
    # write dicts in json files
    dir_path = './data/squad_data'
    if  os.path.isdir(dir_path) == False:
        os.mkdir(dir_path)
    os.chdir(dir_path)

    i=0
    for doc in docs:
        i+=1
        with open(f'./doc{i}.json','w') as outfile:
            json.dump(doc,outfile,ensure_ascii=False).encode('utf-8')
    """
docs = convert_to_haystack_format('./data/plain_data')
print (docs[0])