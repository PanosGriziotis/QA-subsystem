
from typing import List, Dict, Any, Optional, Union

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from haystack.pipelines import Pipeline
from haystack.nodes import  PromptNode, PromptTemplate, AnswerParser, EmbeddingRetriever, SentenceTransformersRanker
from haystack.nodes.base import BaseComponent

from document_store.initialize_document_store import document_store as DOCUMENT_STORE
from utils.data_handling_utils import post_process_generator_answers, remove_second_answers_occurrence

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")


class Generator(BaseComponent):
    """"""
    outgoing_edges = 1

    def __init__(self,
                prompt_messages:List[Dict]=[
                     {"role": "system", "content": 'Χρησιμοποιώντας τις πληροφορίες που περιέχονται στο παρακάτω Κείμενο, δώσε μια ολοκληρωμένη απάντηση στην Ερώτηση. Εάν δεν μπορείς να απαντήσεις με βάση το Κείμενο, απάντα "Δεν γνωρίζω".'},
                     {"role": "user", "content": 'Ερώτηση: {query} | Κείμενο: {join(documents)} | Απάντηση: '}
                     ]):
        
        self.model_name = "ilsp/Meltemi-7B-Instruct-v1"
        self.model = load_model(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt = self.tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
        self.prompt_template = PromptTemplate(prompt = self.prompt,
                                            output_parser=AnswerParser(pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)"))

        super().__init__()

    def run(self, query, documents, max_new_tokens:int=100, temperature:float = 0.4, top_p:float = 0.5):
        """"""
        generation_kwargs={
                            'max_new_tokens': max_new_tokens,
                            'temperature': temperature,
                            'do_sample': True,
                            'top_p': top_p
                            }
        
        generator = PromptNode(model_name_or_path = self.model_name,
                               default_prompt_template = self.prompt_template,
                               top_k= 1,
                               model_kwargs = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'task_name': 'text2text-generation',
            'device': None,
            "generation_kwargs": generation_kwargs
        })


        result, _ = generator.run(query=query, documents=documents)
        
        # Post-process answers to avoid incomplete text resulting from the max_new_tokens parameter
        result = post_process_generator_answers(result)
        result = remove_second_answers_occurrence(result)
        
        return result, 'output_1'
    
    def run_batch(
        self):
         return
    
def load_model (model_name):
    """"""
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,  quantization_config=bnb_config, device_map="auto")
    # Disable Tensor Parallelism 
    model.config.pretraining_tp=1
    
    return model

retriever = EmbeddingRetriever(embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2", document_store=DOCUMENT_STORE)
ranker = SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco")
generator = Generator()

p = Pipeline()
p.add_node(component=retriever, name ="Retriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
p.add_node(component=generator, name="Generator", inputs=["Ranker"])
rag_pipeline = p

if __name__ == "__main__":

    # ============TEST PIPELINE================
    result = rag_pipeline.run(query="Πως μεταδίδεται ο covid;", params={"Retriever": {"top_k":10}, "Ranker": {"top_k":10}, "Generator": {"max_new_tokens": 100}})
    print (result)

