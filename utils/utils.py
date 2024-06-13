from haystack.nodes import TransformersTranslator
import json 
#import spacy
from haystack.nodes import PreProcessor
from haystack import Document
from typing import List
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding
from haystack.nodes import BM25Retriever
from typing import Optional, List
from haystack import Document
import random
import os
from transformers import AutoTokenizer
from haystack.nodes import  PromptNode, PromptTemplate, AnswerParser, PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore

class _MT5inputConverter:
    def __call__(
        self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        conditioned_doc = " " + " ".join([d.content for d in documents])

        # concatenate question and support document into MT5 input
        query_and_docs = " {}<\s>{}<\s>".format(query, conditioned_doc)

        return tokenizer([(query_and_docs)], truncation=True, padding=True, return_tensors="pt")

def train_dev_split(filepath, dev_split):

    with open (filepath, 'r') as file:
        data = json.load(file)['data']
        random.shuffle(data)
        dev_samples = len(data) * dev_split
        train_samples = round(len(data) - dev_samples)
        train_set = data[:train_samples]
        dev_set = data[train_samples:]      
        dev_len = 0
        train_len = 0
        for i, set in enumerate([train_set, dev_set]):
            for par in set:
                for qas in par['paragraphs']:
                    for qas in qas['qas']:
                        if i == 0:
                            train_len += 1
                        else:
                            dev_len += 1
        print (f"train set instances: {train_len}\n dev set instances: {dev_len}")
        path = os.path.dirname (filepath)
        with open(os.path.join(path, "train_file.json"), 'w') as train_file:
            json.dump({'data':train_set}, train_file, ensure_ascii=False, indent=4)
        with open(os.path.join(path,"dev_file.json"), 'w') as dev_file:       
            json.dump ({'data':dev_set}, dev_file, ensure_ascii=False, indent=4)


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

def monitor_directory(directory):
    """delete files when needed"""
    # Get initial list of files in the directory
    initial_files = set(os.listdir(directory))
    
    while True:
        # Get current list of files in the directory
        current_files = set(os.listdir(directory))
        
        # Find new files by subtracting initial files from current files
        new_files = current_files - initial_files
        
        # If there are new files, delete them and store their names
        if new_files:
            for file_name in new_files:
                file_path = os.path.join(directory, file_name)
                # Delete the file
                os.remove(file_path)
                # Store the file name in a variable or data structure
                # For example, you can append it to a list
                print(f"Deleted file: {file_name}")
        
        # Update initial files for the next iteration
        initial_files = current_files
        
        # Sleep for some time before checking again
        time.sleep(1)  # Adjust this value as needed


if __name__ == '__main__': 
    AnswerParser(pattern = r"(?<=<\|assitant\|>\n)([\s\S]*)")