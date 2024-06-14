from typing import List, Optional, Dict, Any, Union, Callable, Tuple
from tqdm import tqdm
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore, ElasticsearchDocumentStore
from haystack.nodes import PreProcessor
from haystack.nodes import FARMReader
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)



def index_eval_labels(document_store, eval_filename: str):
    """
    Index evaluation labels into the document store.

    Parameters:
    document_store (DocumentStore): The document store instance.
    eval_filename (str): The path to the evaluation dataset file.
    """
    label_preprocessor = PreProcessor(
        split_length=256,
        split_respect_sentence_boundary=False,
        clean_empty_lines=False,
        clean_whitespace=False,
    )
 
    document_store.add_eval_data(
        filename=eval_filename,
        doc_index=document_store.index,
        label_index=document_store.label_index,
        preprocessor=label_preprocessor,
    )

def get_eval_labels_and_paths(document_store, tempdir) -> Tuple[List[dict], List[Path]]:
    """
    Retrieve evaluation labels and file paths for documents in the document store.

    Parameters:
    document_store (DocumentStore): The document store instance.
    tempdir: A temporary directory instance for storing document files.

    Returns:
    Tuple[List[dict], List[Path]]: A tuple containing a list of evaluation labels and a list of file paths.
    """
    file_paths = []
    docs = document_store.get_all_documents()

    for doc in docs:
        file_name = f"{doc.id}.txt"
        file_path = Path(tempdir.name) / file_name
        file_paths.append(file_path)
        
        with open(file_path, "w") as f:
            f.write(doc.content)
    
    evaluation_set_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)

    return evaluation_set_labels, file_paths


def evaluate_retriever(retriever, document_store, eval_filename: str, top_k: Optional[int] = None, top_k_list: Optional[List[int]] = None, **kwargs) -> Dict[int, dict]:
    """
    Evaluate a retriever on a SQuAD format evaluation dataset. If a top_k_list is provided, the evaluation is iterative for each top_k value, generating one evaluation report for each value.
    """
    index_eval_labels(document_store, eval_filename)

    if top_k_list is not None:
        reports = {}
        for k in tqdm(top_k_list):
            reports[k] = retriever.eval(label_index=document_store.label_index, doc_index=document_store.index, top_k=k, document_store=document_store)
        return reports
    else:
        if top_k is None:
            top_k = retriever.top_k
        report = retriever.eval(label_index=document_store.label_index, doc_index=document_store.index, top_k=top_k, document_store=document_store)
        return report

def evaluate_reader(reader:FARMReader, eval_filename: str, top_k: Optional[int] = None, top_k_list: Optional[List[int]] = None) -> Dict[int, dict]:
    """
    Evaluate reader on a SQuAD format evaluation dataset.
    """
    data_dir = os.path.dirname(eval_filename)

    if top_k_list is not None:
        reports = {}
        for k in tqdm(top_k_list, desc="Evaluating reader"):
            reader.top_k = k
            reports[k] = reader.eval_on_file(data_dir, eval_filename)
        return reports
    elif top_k is not None:
        reader.top_k = top_k
        return reader.eval_on_file(data_dir, eval_filename)
    else:
        return reader.eval_on_file(data_dir, eval_filename)

def run_experiment(exp_name: str, eval_filename: str, pipeline_path: str, run_name: str, query_params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}):
    """
    Run an experiment with the given parameters.

    exp_name: The name of the experiment.
    eval_filename: The path to the evaluation dataset file.
    pipeline_path: The path to the pipeline YAML file.
    run_name: The name of the experiment run.
    query_params: Parameters for query pipeline
    """

    # Create a temporary directory and document store to get eval data file paths
    temp_dir = tempfile.TemporaryDirectory()
    
    eval_ds = InMemoryDocumentStore()

    index_eval_labels(eval_ds, eval_filename)

    evaluations_set_labels, file_paths = get_eval_labels_and_paths(eval_ds, temp_dir)
    
    # Load pipelines from YAML file
    query_pipeline = Pipeline.load_from_yaml(path=Path(pipeline_path), pipeline_name='query')
    index_pipeline = Pipeline.load_from_yaml(path=Path(pipeline_path), pipeline_name='indexing')
    
    # TODO define the index in the yaml pipelines for evaluation (?). Build a seperate index pipeline
    # Get document store from index pipeline and delete existing documents
    
    # Execute experiment run
    Pipeline.execute_eval_run(
        index_pipeline=index_pipeline,
        query_pipeline=query_pipeline,
        evaluation_set_labels=evaluations_set_labels,
        corpus_file_paths=file_paths,
        experiment_name=exp_name,
        experiment_run_name=run_name,
        evaluation_set_meta=os.path.basename(eval_filename),
        add_isolated_node_eval=True,
        experiment_tracking_tool="mlflow",
        experiment_tracking_uri="http://localhost:5000",
        query_params=query_params,
        sas_model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        sas_batch_size=32,
        sas_use_gpu=True,
        reuse_index=False
    )
    
    temp_dir.cleanup()
    