import json
import os
import argparse
import requests
from haystack.nodes import FARMReader, BM25Retriever, EmbeddingRetriever, DensePassageRetriever, SentenceTransformersRanker
from haystack.document_stores import ElasticsearchDocumentStore
from main import evaluate_reader, evaluate_retriever_ranker_pipeline, fetch_eval_dataset
from utils import load_and_save_npho_datasets, load_and_save_xquad_dataset

def evaluate_on_xquad(eval_type):
    load_and_save_xquad_dataset()
    if eval_type == "reader":
        run_readers_evaluation("datasets/xquad-el.json")
    elif eval_type == "retriever":
        run_retriever_evaluation("datasets/xquad-el.json")

def evaluate_on_npho(version, eval_type):
    load_and_save_npho_datasets()
    if eval_type == "reader":
        run_readers_evaluation(f"datasets/npho-covid-SQuAD-el_{version}.json")
    elif eval_type == "retriever":
        run_retriever_evaluation(f"datasets/npho-covid-SQuAD-el_{version}.json")

def evaluate_on_other_dataset(file_path, eval_type):
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return
    if eval_type == "reader":
        run_readers_evaluation(file_path)
    elif eval_type == "retriever":
        run_retriever_evaluation(file_path)

# Reader evaluation
def run_readers_evaluation(eval_filename):
    readers = [
        "panosgriz/xlm-roberta-squad2-covid-el_small",
        "panosgriz/xlm-roberta-squad2-covid_el",
        "panosgriz/mdeberta-v3-base-squad2-covid-el",
        "panosgriz/mdeberta-v3-base-squad2-covid-el_small",
        "panosgriz/xlm-roberta-covid-small-early-stopping",
    ]
    readers = [FARMReader(model_name_or_path=reader) for reader in readers]
    reports = {}
    for reader in readers:
        reports[reader.model_name_or_path] = evaluate_reader(reader, eval_filename)
    with open(f"reports/report_readers_{os.path.basename(eval_filename).split('.')[0]}.json", "w") as fp:
        json.dump(reports, fp, ensure_ascii=False, indent=4)

# Retriever evaluation
def run_retriever_evaluation(eval_filename):
    top_k_values = [x for x in range(1, 51)]
    results = {}
    ranker = SentenceTransformersRanker(
        model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco"
    )
    retriever_configs = {
        "bm25": {
            "document_store": {
                "embedding_dim": 384,
                "label_index": "label_index",
                "index": "eval_docs"
            },
            "retriever_class": BM25Retriever,
            "retriever_args": {}
        },
        "embedding_retriever": {
            "document_store": {
                "embedding_dim": 384,
                "label_index": "label_index",
                "index": "eval_docs"
            },
            "retriever_class": EmbeddingRetriever,
            "retriever_args": {
                "embedding_model": "/home/pgriziotis/thesis/qa-subsystem/dev/retriever/adapted_retriever"
            }
        },
        "dpr": {
            "document_store": {
                "embedding_dim": 768,
                "label_index": "label_index",
                "index": "eval_docs",
            },
            "retriever_class": DensePassageRetriever,
            "retriever_args": {
                "query_embedding_model": "panosgriz/bert-base-greek-covideldpr-query_encoder",
                "passage_embedding_model": "panosgriz/bert-base-greek-covideldpr-ctx_encoder",
                "max_seq_len_passage": 300
            }
        }
    }
    for retriever_type, config in retriever_configs.items():
        requests.delete("http://localhost:9200/*")
        ds = ElasticsearchDocumentStore(**config["document_store"])
        retriever = config["retriever_class"](document_store=ds, **config["retriever_args"])
        reports = evaluate_retriever_ranker_pipeline(
            retriever=retriever,
            ranker=ranker,
            document_store=ds,
            eval_filename=eval_filename,
            top_k_list=top_k_values
        )
        results[f"{retriever_type}_&_ranker"] = reports
    output_file = f"reports/retrievers_eval_report_{os.path.basename(eval_filename)}"
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--evaluate", choices=['xquad', 'npho_10', 'npho_20', 'other'], required=True,
                        help="Choose the dataset to evaluate on")
    parser.add_argument("--eval_type", choices=['reader', 'retriever'], required=True,
                        help="Choose the type of evaluation to run")
    parser.add_argument("--file_path", type=str, help="File path for 'other' dataset evaluation")

    args = parser.parse_args()

    if args.evaluate == "xquad":
        evaluate_on_xquad(args.eval_type)
    elif args.evaluate == "npho_10":
        evaluate_on_npho('10', args.eval_type)
    elif args.evaluate == "npho_20":
        evaluate_on_npho('20', args.eval_type)
    elif args.evaluate == "other":
        if args.file_path:
            evaluate_on_other_dataset(args.file_path, args.eval_type)
        else:
            print("Please provide a file path for 'other' dataset evaluation.")

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    main()