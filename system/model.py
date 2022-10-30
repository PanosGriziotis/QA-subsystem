# haystack class module

from pathlib import Path
from typing import List
from haystack.nodes import TextConverter
from haystack.nodes import PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.nodes import FARMReader, TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
import os
from haystack.utils import print_answers

class QuestionAnsweringSystem():

  def __init__(self, data_dir):
      
      self.data_dir = data_dir
      self.docs = self.convert_to_haystack_format()
      self.ds = self.create_document_store(self.docs)

  def convert_to_haystack_format(self):
    
    # convert txt files to dicts
    all_docs = []
    converter = TextConverter(valid_languages=["el"])
    for file in Path(self.data_dir).iterdir():
        all_docs.append(converter.convert(file_path=file, meta=None)[0])
    # clean and split
    preprocessor = PreProcessor(
        language='el',
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=180,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(all_docs)
    return docs
  
  def create_document_store(self, docs, similarity_metric='cosine_similarity'):
    document_store = InMemoryDocumentStore(similarity_metric)
    try:
      document_store.write_documents(docs)
      return document_store
    except Exception as e:
      print(e)

  def fine_tune_reader_model (self, save_dir, train_filename, num_epochs=2, batch_size=2, use_gpu=True, model="deepset/xlm-roberta-base-squad2"):
      
      reader = FARMReader(model_name_or_path = model)
      try:
        reader.train(
            data_dir = self.data_dir,
            train_filename = train_filename,
            use_gpu = use_gpu,
            batch_size=batch_size,
            n_epochs= num_epochs,
            max_seq_len = 384,
            save_dir = save_dir,
            num_processes = 1,
            )
        print ('fine-tuning done')
      except Exception as e:
        print (e)

  def get_retriever(self):
    return TfidfRetriever(self.ds)

  def get_reader(self, trained_model_path):
    return FARMReader(model_name_or_path=trained_model_path)

  def get_pipeline (self, reader, retriever):
    return ExtractiveQAPipeline(reader, retriever)

  def get_answers (seld, query, pipeline, top_k_retriever=10, top_k_reader = 1):
    predictions = pipeline.run(
      query=query, params={"Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
      )
    return predictions

if __name__ == '__main__':

  DATA_DIR = '../data/documents'
  TRAIN_FILE = 'eody_train_data.json'
  MODEL_PATH = './NEW_model'

  if os.path.isdir(MODEL_PATH) == False:
      os.mkdir(MODEL_PATH)

  # initialize class
  qa_system = QuestionAnsweringSystem(DATA_DIR)

  # fine-tune model on data
  #qa_system.fine_tune_reader_model (MODEL_PATH,TRAIN_FILE, num_epochs = 2, batch_size=2)
  #print ('fine-tuning done')

  #retriever & reader

  
  retriever = qa_system.get_retriever()

  reader = qa_system.get_reader(MODEL_PATH)
  # pipeline 
  pipe = qa_system.get_pipeline(reader, retriever)

  print_answers (qa_system.get_answers("Μπορώ να ταξιδέψω χωρίς εμβόλιο;",pipe), details = 'minimum')

  print (' ')
  print ('eody model')