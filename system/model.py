# haystack class module

from pathlib import Path
from typing import List
from haystack.nodes import TextConverter
from haystack.nodes import PreProcessor
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline

class QuestionAnsweringSystem():

  def __init__(self, data_dir, train_filename):
      
      self.data_dir = data_dir
      self.train_filename = train_filename

  def convert_to_haystack_format(self) -> List:
    
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
        split_length=200,
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

  def fine_tune_reader_model (self, reader_model_path, n_epochs, model="deepset/xlm-roberta-large-squad2", use_gpu=True):

      reader = FARMReader(model_name_or_path=model, use_gpu=use_gpu)
      try:

        reader.train(
            data_dir = self.data_dir,
            train_filename=self.train_filename,
            use_gpu=use_gpu,
            n_epochs=n_epochs,
            save_dir = reader_model_path)
        print ('fine-tuning done')
      except Exception as e:
        print (e)

  def get_retriever(self, document_store):
    return BM25Retriever(document_store)
  def get_reader(self, reader_model_path):
    return FARMReader(model_name_or_path=reader_model_path)
  def get_pipeline (self, reader, retriever):
    return ExtractiveQAPipeline(reader, retriever)
  def get_answers (seld, query, pipeline, top_k_retriever=10, top_k_reader = 3):
    predictions = pipeline.run(
      query=query, params={"Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
      )
    return [{'answer': result['answer'], 'context': result['context'], 'startLoc': result['offset_start_in_doc'], 'endLoc': result['offset_end_in_doc'], 'docText':  self.get_haystack_doc_text_by_id(document_store, result['document_id']), 'probability': result['probability']} for
                result in predictions['answers']]