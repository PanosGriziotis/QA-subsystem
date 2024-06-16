
from haystack.nodes import  PromptNode, PromptTemplate, AnswerParser, PreProcessor
from haystack.pipelines import Pipeline
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import SquadData
from haystack.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from haystack.document_stores import ElasticsearchDocumentStore
from typing import List, Dict

#model = AutoModelForCausalLM.from_pretrained("ilsp/Meltemi-7B-Instruct-v1",  load_in_4bit=True, device_map="auto")
#tokenizer = AutoTokenizer.from_pretrained("ilsp/Meltemi-7B-Instruct-v1")

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
    generator = PromptNode(model_name_or_path="ilsp/Meltemi-7B-Instruct-v1",
                            default_prompt_template = prompt_template,
                            top_k=1,
                            model_kwargs = {'model': model,
                                            'tokenizer': tokenizer,
                                            'task_name': 'text2text-generation',
                                            'device': None, 
                                            "generation_kwargs": {'max_new_tokens':max_new_tokens,'do_sample':True, 'temperature': temperature, 'top_p':top_p}})
    return generator

def create_rag_pipeline (retriever, generator):
    """"""

    p = Pipeline()
    p.add_node(component=retriever, name="retriever", inputs=["Query"])
    p.add_node(component=generator, name="prompt_node", inputs=["retriever"])
    
    return p

if __name__ == "__main__":

    
    # Initialize prompt in message format
    messages = [
        {"role": "system", "content": "Χρησιμοποιώντας τις πληροφορίες που περιέχονται στο Κείμενο, δώσε μια ολοκληρωμένη απάντηση στην Ερώτηση. Εάν η απάντηση δεν μπορεί να συναχθεί από το Κείμενο, απάντα 'Δεν ξέρω'."},
        {"role": "user", "content": "Κείμενο: {join(documents)}; Ερώτηση: {query}"},
    ]
    
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="wiki")
    retriever = BM25Retriever(document_store = document_store, top_k=2)

    generator = build_generator(messages=messages)

    pipe = create_rag_pipeline(retriever=retriever, generator=generator)

    result=pipe.run(query="Πόσες μέρες πρέπει να μείνω σε καραντίνα;")

    print (result["answers"][0].answer)

