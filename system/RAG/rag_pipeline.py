
from haystack.nodes import  PromptNode, PromptTemplate, AnswerParser, PreProcessor
from haystack.pipelines import Pipeline
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import SquadData
from haystack.schema import Document
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

#model = AutoModelForCausalLM.from_pretrained("ilsp/Meltemi-7B-Instruct-v1",  load_in_4bit=True, device_map="auto")
#tokenizer = AutoTokenizer.from_pretrained("ilsp/Meltemi-7B-Instruct-v1")

from haystack.document_stores import ElasticsearchDocumentStore

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="wiki")
retriever = BM25Retriever(document_store=document_store, top_k=2)
"""
with open ("/home/pgriziotis/thesis/QA-subsystem/data/npho/npho_squad_20tokens.json", "r") as fp:
    data = json.load(fp)
    data = SquadData(data)
    paragraphs = data.get_all_paragraphs()

preprocessor = PreProcessor(
        split_by = "word",
        split_length=300,
        split_respect_sentence_boundary=True,
        clean_empty_lines=False,
        clean_whitespace=False,
        language = "el"
    )

documents =  [Document (content= paragraph) for paragraph in paragraphs]
documents = preprocessor.process (documents)
document_store.write_documents(documents)
"""
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Quantization config
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

# Initialize model
model = AutoModelForCausalLM.from_pretrained("ilsp/Meltemi-7B-Instruct-v1",  quantization_config=bnb_config, device_map="auto")

# Disableb Tensor Parallelism 

model.config.pretraining_tp=1 

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("ilsp/Meltemi-7B-Instruct-v1")

# Initialize prompt in message format
messages_1 = [
    {"role": "system", "content": "Χρησιμοποιώντας τις πληροφορίες που περιέχονται στο Κείμενο, δώσε μια ολοκληρωμένη απάντηση στην Ερώτηση. Εάν η απάντηση δεν μπορεί να συναχθεί από το Κείμενο, απάντα 'Δεν ξέρω'."},
    {"role": "user", "content": "Κείμενο: {join(documents)}; Ερώτηση: {query}"},
]

# convert prompt to correct zephyr format
prompt = tokenizer.apply_chat_template(messages_1, add_generation_prompt=True, tokenize=False)

prompt_template = PromptTemplate(prompt = prompt,
                                 output_parser=AnswerParser(pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)"))

generator = PromptNode(model_name_or_path="ilsp/Meltemi-7B-Instruct-v1",
                        default_prompt_template = prompt_template,
                        model_kwargs = {'model': model,
                                        'tokenizer': tokenizer,
                                        'task_name': 'text2text-generation',
                                        'device': None, 
                                        "generation_kwargs": {'max_new_tokens':100,'do_sample':True, 'temperature': 0.4, 'top_p':0.5}}
)

pipe = Pipeline()
pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
pipe.add_node(component=generator, name="prompt_node", inputs=["retriever"])

result=pipe.run(query="Πόσες μέρες πρέπει να μείνω σε καραντίνα;")

print (result)