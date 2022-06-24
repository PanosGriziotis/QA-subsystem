# main module
import os
from model import QuestionAnsweringSystem


DATA_DIR = '../data/first_model_data'
TRAIN_FILE = 'answers.json'
MODEL_PATH = './models'

if os.path.isdir(MODEL_PATH) == False:
    os.mkdir(MODEL_PATH)

# initialize class
qa_system = QuestionAnsweringSystem(DATA_DIR, TRAIN_FILE)

# get documents
docs =  qa_system.convert_to_haystack_format()
print ("pre-processing done")

# document store
ds = qa_system.create_document_store(docs)
print ('created document store')

# fine-tune model on data
qa_system.fine_tune_reader_model (MODEL_PATH,n_epochs=1)
print ('fine-tuning done')

# retriever & reader
retriever = qa_system.get_retriever(ds)
reader = qa_system.get_reader(MODEL_PATH)

"""
# pipeline 
pipe = qa_system.get_pipeline(reader, retriever)

# get answers dictionary
answers  = qa_system.get_answers("",pipe)
print(answers)
"""