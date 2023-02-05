from haystack.nodes import FARMReader
import os
from haystack.utils import EarlyStopping
from haystack.nodes.label_generator import PseudoLabelGenerator
from typing import List

def fine_tune_reader_model (data_dir, train_filename,  save_dir, use_gpu=True):

    reader = FARMReader(model_name_or_path = "deepset/xlm-roberta-base-squad2")
    try:
      reader.train(
          data_dir = data_dir,
          train_filename = train_filename,
          use_gpu = use_gpu,
          n_epochs= 10,
          batch_size= 12,
          max_seq_len = 384,
          num_processes = 1,
          )
      print (f'fine-tuning done. Model saved in directory: {save_dir}')
    except Exception as e:
      print (e)

def fine_tune_dense_retriever(document_store, retriever):
    query_doc_pairs  = []
    with open ('data/who/queries1.el.txt', 'r', encoding='utf-8') as q_file:
        with open ('data/who/answers1.el.txt', 'r', encoding='utf-8') as a_file:
            queries = q_file.readlines()
            docs = a_file.readlines()
        for q, d in zip (queries,docs):
            query_doc_pairs.append({"question": q , "document": d})

    psg = PseudoLabelGenerator(query_doc_pairs, retriever, cross_encoder_model_name_or_path="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  batch_size=32, top_k=10)
    output, pipe_id = psg.run(documents=document_store.get_all_documents(index="documents"))
    retriever.train(output["gpl_labels"])
    retriever.save("adapted_retriever")
    
if __name__ == '__main__':
  save_dir = './model'
  if not os.path.isdir ('./model'):
    os.mkdir('./model')

  fine_tune_reader_model(data_dir = '../data/eody', train_filename= 'train.json', save_dir='./model')
