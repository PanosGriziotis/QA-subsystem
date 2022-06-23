from encodings import utf_8
from haystack.nodes import TextConverter
from pathlib import Path
from haystack.nodes import PreProcessor
import spacy
from haystack.utils.preprocessing import convert_files_to_dicts


def truecase_text(text):
    "Applies a truecaser to text based on name entities"
    spacy_nlp = spacy.load("el_core_news_sm")
    text = spacy_nlp(text)
    words = [word.text for word in text]
    ents = [ent.text for ent in text.ents]
    capitalized_words = [w.capitalize() if w in ents else w for w in words]
    true_cased_words = [w.lower() if w.isupper() else w for w in capitalized_words]
    return ' '.join(true_cased_words)
