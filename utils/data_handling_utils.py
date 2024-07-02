from typing import List

from haystack.nodes import TransformersTranslator
import json 

from haystack.nodes import PreProcessor
from haystack import Document

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from haystack import Document
import random
import os
from transformers import AutoTokenizer
from haystack.nodes import PreProcessor
import re
import nltk
from haystack.utils import SquadData


def post_process_generator_answers (result):
    """Post-process answers in the Generator's output to avoid incomplete text resulting from the max_new_tokens parameter."""
    
    processed_answers = []
    for answer_obj in result ["answers"]:
        answer_obj.answer = remove_incomplete_sentences(answer_obj.answer)
        processed_answers.append(answer_obj)

    result['answers'] = processed_answers
    return result

def remove_incomplete_sentences(text):
    """Filter out incomplete sentences from a text where sentences might be cut off"""

    nltk.download('punkt')

    sentences = nltk.sent_tokenize(text)
    complete_sentences = []

    sentence_end_regex = re.compile(r'.*[\.\;!]$')

    for sentence in sentences:
        
        if sentence_end_regex.match(sentence.strip()):
            complete_sentences.append(sentence)

    filtered_text = ' '.join(complete_sentences)

    return filtered_text

def remove_second_answers_occurrence(result):
    new_result= {}
    key_count = 0

    for key, value in result.items():
        if key == "answers":
            key_count += 1
            if key_count == 2:
                continue  # Skip the second occurrence
        new_result[key] = value

    return new_result

def split_squad_dataset (filepath, split_ratio: int = 0.1):

    with open(filepath, encoding="utf-8") as f:

        # load and count total num of examples
        data = json.load(f)
        num_of_examples =  SquadData (data).count()

        # shuffle examples
        data = data["data"]
        random.shuffle(data)
        counter = 0
        test_set = []

        for article in data:
            
            for paragraph in article["paragraphs"]:
                counter += (len(paragraph["qas"]))

            if counter >= round (num_of_examples * split_ratio):
                break
            else:
                test_set.append (article)
                    
        train_set = {"data" : data [len(test_set):]}
        test_set = {"data" : test_set}

    print (f"train set instances: {(num_of_examples-counter)}\n dev set instances: {counter}")
    
    # Write datasets
    path = os.path.dirname (filepath)

    with open(os.path.join(path, "train_file.json"), 'w') as train_file:
        json.dump(train_set, train_file, ensure_ascii=False, indent=4)

    with open(os.path.join(path,"dev_file.json"), 'w') as dev_file:       
        json.dump (test_set, dev_file, ensure_ascii=False, indent=4)


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


def is_english(text):
    DetectorFactory.seed = 0
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def remove_english_text(lines):
    non_english_lines = []
    for line in lines:
        if not is_english(line):
            non_english_lines.append(line)
    return non_english_lines
"""
def get_true_case (text):
    spacy_nlp = spacy.load("el_core_news_sm")
    words = [word.text for word in spacy_nlp(text)]
    ents = [ent.text for ent in spacy_nlp(text).ents]
    capitalized_words = [w.capitalize() if w in ents else w for w in words]
    true_cased_words = [w.lower() if w.isupper() else w for w in capitalized_words]
    return ' '.join (join_punctuation(true_cased_words))
"""

def fit_passage_in_max_len (context, answer, max_seq_len_passage):
    """This function helps trimming a given passage in order to not exceed the maximum sequence length of a model while ensuring to also include the respective answer"""

    # Step 1: tokenize both context and answer with model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
    context_tokens = tokenizer.tokenize (context)
    answer_tokens = tokenizer.tokenize (answer)

    # Step 2: Calculate total tokens to keep
    tokens_to_keep = max_seq_len_passage - len(answer_tokens) - 3  # Reserve tokens for [CLS], [SEP], and space

    # Step 3: Validate if the given context is indeed exceeding the given max_seq_len else trimm the context
    if len(context_tokens) <= tokens_to_keep:


        return tokenizer.convert_tokens_to_string (answer_tokens), tokenizer.convert_tokens_to_string (context_tokens)
    else:
        try:
            # Find the token number in which the answer starts
            answer_start = context_tokens.index(answer_tokens[0])

            # Calculate context window
            context_start = max(0, answer_start - (tokens_to_keep // 2))
            context_end = min(len(context_tokens), answer_start + len(answer_tokens) + (tokens_to_keep // 2))

            # Adjust context window if needed
            if context_end - context_start < tokens_to_keep:
                if context_end == len(context_tokens):
                    context_start = max(0, context_end - tokens_to_keep)
                else:
                    context_end = min(len(context_tokens), context_start + tokens_to_keep)

            # Trim context, while including the answer
            trimmed_context_tokens = context_tokens[context_start:context_end]
            trimmed_context = tokenizer.convert_tokens_to_string(trimmed_context_tokens)
            answer_new = tokenizer.convert_tokens_to_string (answer_tokens)
            
            return  answer_new, trimmed_context
        
        except ValueError:
            print ("Answer not in context")

def get_query_doc_pairs_from_dpr_file (dpr_filename):
    
    query_doc_pairs = []
    with open (dpr_filename, "r") as fp:
        data = json.load(fp)
        for d in data:
            question = d["question"]
            for positive_ctx in d["positive_ctxs"]:
                document = positive_ctx["text"]
        query_doc_pairs.append({"question": question, "document": document})
    return query_doc_pairs


def is_english(text):
    DetectorFactory.seed = 0
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False