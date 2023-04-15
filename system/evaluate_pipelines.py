from haystack.document_stores import ElasticsearchDocumentStore
import os
from haystack.nodes import PreProcessor
from haystack.nodes import BM25Retriever
from haystack import Pipeline
import tempfile
from pathlib import Path
from haystack.nodes import PreProcessor
from haystack.document_stores import InMemoryDocumentStore
from pipelines import create_pipelines

class Evaluator ():
    def __init__(self) -> None:
        self.query_pipeline = self.query_pipeline
        self.document_store = self.document_store    
    
    def index_eval_data(self,
                        label_index,
                        doc_index,
                        filename,
                        preprocessor
    ):

        # Eval data must be in SQuAD format. A new ds instance must be created with eval indeces
        self.document_store.delete_documents(index=doc_index)
        self.document_store.delete_documents(index=label_index)

        # The add_eval_data() method converts the given dataset in json format into Haystack document and label objects. Those objects are then indexed in their respective document and label index in the document store. The method can be used with any dataset in SQuAD format.
        self.document_store.add_eval_data(filename, doc_index, label_index, preprocessor)

    def evaluate_retriever (self, top_k, label_index, doc_index):
        retriever = self.query_pipeline.get_node("Retriever")
        # if retriever uses DENSE vectors:
        self.document_store.update_embeddings(retriever, index=doc_index)
        return retriever.eval(top_k=top_k, label_index=label_index, doc_index=doc_index)
    
    def evaluate_reader (self, label_index, doc_index):
        reader = self.query_pipeline.get_node ("Reader")
        return reader.eval(document_store=self.document_store, label_index=label_index, doc_index=doc_index)

    def evaluate_qa_pipeline (self, params):
        eval_labels = self.document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)
        self.query_pipeline.eval(labels=eval_labels, params= params)


# EXPERIMENTS ON MLFLOW


def index_eval_data(
    filename,
    label_preprocessor
):
    document_store = InMemoryDocumentStore()
    document_store.add_eval_data(
    filename=filename,
    doc_index=document_store.index,
    label_index=document_store.label_index,
    preprocessor=label_preprocessor,
)
    evaluation_set_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)
    docs = document_store.get_all_documents()
    temp_dir = tempfile.TemporaryDirectory()
    file_paths = []
    for doc in docs:
        file_name = doc.id + ".txt"
        file_path = Path(temp_dir.name) / file_name
        file_paths.append(file_path)
        with open(file_path, "w") as f:
            f.write(doc.content)
    file_metas = [d.meta for d in docs]
    return file_paths, file_metas, evaluation_set_labels



# The add_eval_data() method converts the given dataset in json format into Haystack document and label objects.
# Those objects are then indexed in their respective document and label index in the document store.
# The method can be used with any dataset in SQuAD format.
# We only use it to get the evaluation set labels and the corpus files.


# the evaluation set to evaluate the pipelines on

# Pipelines need files as input to be able to test different preprocessors.
# Even though this looks a bit cumbersome to write the documents back to files we gain a lot of evaluation potential and reproducibility.


EXPERIMENT_NAME = "sparse retrieval"

sparse_eval_result = Pipeline.execute_eval_run(
    index_pipeline=index_pipeline,
    query_pipeline=query_pipeline,
    evaluation_set_labels=evaluation_set_labels,
    corpus_file_paths=file_paths,
    corpus_file_metas=file_metas,
    experiment_name=EXPERIMENT_NAME,
    experiment_run_name="sparse",
    corpus_meta={"name": "nq_dev_subset_v2.json"},
    evaluation_set_meta={"name": "nq_dev_subset_v2.json"},
    pipeline_meta={"name": "sparse-pipeline"},
    add_isolated_node_eval=True,
    experiment_tracking_tool="mlflow",
    experiment_tracking_uri="https://public-mlflow.deepset.ai",
    reuse_index=True,
)


#if __name__ == '__main__':


#document_store = ElasticsearchDocumentStore(
    #index=doc_index,
    #label_index=label_index,
    #embedding_field="emb",
    #embedding_dim=768,
    #excluded_meta_data=["emb"]
label_preprocessor = PreProcessor(
    split_length=200,
    split_overlap=0,
    split_respect_sentence_boundary=False,
    clean_empty_lines=False,
    clean_whitespace=False,
)