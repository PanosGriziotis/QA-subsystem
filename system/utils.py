from haystack.nodes import TransformersTranslator
import json 
import spacy
from haystack.nodes import PreProcessor
from haystack.nodes import TextConverter
from haystack import Document
from typing import List
import textract
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding, MT5Tokenizer
from typing import Optional, List
from haystack import Document

class _MT5inputConverter:
    def __call__(
        self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        conditioned_doc = " " + " ".join([d.content for d in documents])

        # concatenate question and support document into MT5 input
        query_and_docs = " {}<\s>{}<\s>".format(query, conditioned_doc)

        return tokenizer([(query_and_docs)], truncation=True, padding=True, return_tensors="pt")


def translate_docs (docs:List[str], use_gpu:bool=False):
    
    max_seq_len = 512
    translator = TransformersTranslator(model_name_or_path="facebook/nllb-200-distilled-600M", use_gpu=use_gpu, max_seq_len=max_seq_len)
    #c_docs = clean_and_split_docs(docs, max_seq_len=max_seq_len)
    try:
        t_docs = translator.translate_batch(documents=docs)[0]
    except AttributeError:
        t_docs = ['<ukn>']
    return t_docs

def split_and_translate (docs:List[str], max_seq_len:int = 512):
    

    preprocessor = PreProcessor (language='en', split_by='word', split_length=max_seq_len,  split_respect_sentence_boundary=True, progress_bar=False)
    idx_to_splitted_docs = {}

    for idx, doc in enumerate (docs):
        tokens = [word for word in doc.strip().split(' ')]
        if len(tokens) > max_seq_len:
            docs.pop(idx)
            doc = Document (content=doc)
            splitted_docs = [doc.content for doc in preprocessor.process([doc])]
            # keep track splitted long answers 
            idx_to_splitted_docs[idx] = splitted_docs
        else:
            continue
    
    t_docs = translate_docs(docs)

    if idx_to_splitted_docs:

        for idx, splitted_docs in idx_to_splitted_docs.items():
            # translate splitted docs and join them 
            t_answer = ' '.join (translate_docs(splitted_docs))
            t_docs.insert(idx, t_answer)

    return t_docs


def join_punctuation(seq, characters='.,;?!:'):
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current
    return ' '.join(seq)


def get_true_case (text):
    spacy_nlp = spacy.load("el_core_news_sm")
    words = [word.text for word in spacy_nlp(text)]
    ents = [ent.text for ent in spacy_nlp(text).ents]
    capitalized_words = [w.capitalize() if w in ents else w for w in words]
    true_cased_words = [w.lower() if w.isupper() else w for w in capitalized_words]
    return ' '.join (join_punctuation(true_cased_words))

def parse_pdf(response):
    tempfile = NamedTemporaryFile(suffix='.pdf')
    tempfile.write(response.content)
    tempfile.flush()
    extracted_data = textract.process(tempfile.name)
    extracted_data = extracted_data.decode('utf-8')
    tempfile.close()



    #print (c_answers)
    #print ('max_len_before:', len(max(answers, key=len)))
    #print ('max_len_after:', len(max(c_answers, key=len)))
    #print ('docs before splitting:', len(answers))
    #print ('docs after splitting:', len(c_answers))

