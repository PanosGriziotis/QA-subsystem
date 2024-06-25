from typing import List, Dict, Any, Optional, Union

from haystack.pipelines import Pipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import  PromptNode, PromptTemplate, AnswerParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from haystack.nodes.base import BaseComponent
from document_store.initialize_document_store import document_store as DOCUMENT_STORE

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")

# TODO add default prompt in a global param or in a file + file with generator params
# TODO add log for checking if given index has documents

class Generator(BaseComponent):
    """"""
    outgoing_edges = 1

    def __init__(self,
                prompt_messages:List[Dict]=[
                     {"role": "system", "content": 'Χρησιμοποιώντας τις πληροφορίες που περιέχονται στο παρακάτω Κείμενο, δώσε μια ολοκληρωμένη απάντηση στην Ερώτηση. Εάν δεν μπορείς να απαντήσεις με βάση το Κείμενο, απάντα "Δεν γνωρίζω".'},
                     {"role": "user", "content": 'Ερώτηση: {query} | Κείμενο: {join(documents)}'}
                     ],
                top_k = 1,
                max_new_tokens:int = 100,
                temperature:int = 0.4,
                top_p:int = 0.5):
        

        model_name = "ilsp/Meltemi-7B-Instruct-v1"
        model = load_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompt = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
        prompt_template = PromptTemplate(prompt = prompt,
                                         output_parser=AnswerParser(pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)"))

        self.generator = PromptNode(model_name_or_path = model_name,
                                    default_prompt_template = prompt_template,
                                    top_k = top_k,
                                    model_kwargs = {'model': model,
                                                    'tokenizer': tokenizer,
                                                    'task_name': 'text2text-generation',
                                                    'device': None, 
                                                    "generation_kwargs": {
                                                        'max_new_tokens': max_new_tokens,
                                                        'do_sample':True,
                                                        'temperature': temperature,
                                                        'top_p':top_p}
                                                    })

        super().__init__()

    def run(self, query, documents):
        """"""
        result, _ = self.generator.run(query=query, documents=documents)
        
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

def rag_pipeline ():
    """"""
    # TODO SET DEFAULT PARAMS

    global DOCUMENT_STORE

    retriever = BM25Retriever(document_store=DOCUMENT_STORE, top_k=2)
    generator = Generator()

    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name ="Retriever", inputs=["Query"])
    pipeline.add_node(component=generator, name="Generator", inputs=["Retriever"])

    return pipeline

if __name__ == "__main__":

    # ==========TEST GENERATOR=================

    #generator= Generator()
    #docs = DS.get_all_documents()
    #result, _ = generator.run(query="Τι πρέπει να κάνει το σχολείο;", documents=docs[:2])
    #result = result["generator_answer"].answer
    #print (result)

    # ============TEST PIPELINE================
    result = rag_pipeline().run(query="Πως μεταδίδεται ο covid;")
    
    print (result["answers"][0].answer)

