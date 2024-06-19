from haystack.pipelines import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from haystack.nodes import  PromptNode, PromptTemplate, AnswerParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from haystack.nodes.base import BaseComponent

from typing import List, Dict, Any, Optional, Union

import os 

host = os.getenv("DOCUMENTSTORE_PARAMS_HOST", "localhost")
port = os.getenv("DOCUMENTSTORE_PARAMS_PORT", 9200)
class Generator(BaseComponent):
    outgoing_edges = 1

    def __init__(self, prompt_messages=None, max_new_tokens=None, temperature=None, top_p=None):
        
        self.generator = build_generator(messages=prompt_messages, 
                                         max_new_tokens=max_new_tokens, 
                                         temperature=temperature, 
                                         top_p=top_p)
        super().__init__()

    def run(self, query, documents):

        result, _ = self.generator.run(query=query, documents=documents)
        answer = result['answers'][0].answer
        output = {"generator_answer": answer}
        return output, 'output_1'
    
    def run_batch(
        self):
         return
    
def load_model ():
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained("ilsp/Meltemi-7B-Instruct-v1",  quantization_config=bnb_config, device_map="auto")

    # Disable Tensor Parallelism 
    model.config.pretraining_tp=1
    
    return model

def build_generator (messages:List[Dict],  max_new_tokens:int = 100, temperature:float = 0.4, top_p:float = 0.5):
    """Build Generator node in RAG pipeline
    
    messages: chat prompt
    max_new_tokens:
    temperature:
    top_p:
    """

    model = load_model()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ilsp/Meltemi-7B-Instruct-v1")
    
    # convert prompt to correct zephyr format
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # create prompt template
    prompt_template = PromptTemplate(prompt = prompt,
                                 output_parser=AnswerParser(pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)"))

    # initialize generator node
    generator = PromptNode( model_name_or_path = "ilsp/Meltemi-7B-Instruct-v1",
                            default_prompt_template = prompt_template,
                            top_k = 1,
                            model_kwargs = {'model': model,
                                            'tokenizer': tokenizer,
                                            'task_name': 'text2text-generation',
                                            'device': None, 
                                            "generation_kwargs": {'max_new_tokens':max_new_tokens,'do_sample':True, 'temperature': temperature, 'top_p':top_p}})
    return generator

# BUILD PIPELINE
retriever = BM25Retriever(document_store=ElasticsearchDocumentStore(host=host, port=port, username="", password="", index="test_api"))

generator = Generator(
    prompt_messages=[
        {"role": "system", "content": 'Χρησιμοποιώντας τις πληροφορίες που περιέχονται στο παρακάτω Κείμενο, δώσε μια ολοκληρωμένη απάντηση στην Ερώτηση. Εάν δεν μπορείς να απαντήσεις με βάση το Κείμενο, απάντα "Δεν γνωρίζω".'},
        {"role": "user", "content": 'Ερώτηση: {query}; Κείμενο: {join(documents)}'}
    ],
    max_new_tokens=100,
    temperature=0.4,
    top_p=0.5    
)

rag_pipeline = Pipeline()


rag_pipeline.add_node(component=retriever, name = "Retriever", inputs=["Query"])
rag_pipeline.add_node(component=generator, name="Generator", inputs=["Retriever"])

if __name__ == "__main__":

    result = rag_pipeline.run(query="Πώς κολλάει ο covid-19;")
    print (result)