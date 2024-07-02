import os
import json
import argparse

from haystack.nodes import BM25Retriever, EmbeddingRetriever, DensePassageRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import SentenceTransformersRanker

from main import evaluate_retriever_ranker_pipeline
from utils import load_and_save_npho_datasets, load_and_save_xquad_dataset

import requests

def evaluate_on_xquad():
    load_and_save_xquad_dataset()
    run_retriever_evaluation_test("datasets/xquad-el.json")

def evaluate_on_npho(version):
    load_and_save_npho_datasets()
    run_retriever_evaluation_test(f"datasets/npho-covid-SQuAD-el_{version}.json")

def evaluate_on_other_dataset(file_path):
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return
    run_retriever_evaluation_test(file_path)

def run_retriever_evaluation_test(eval_filename):
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

        # evaluate with ranker
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
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--file_path", type=str, help="File path for 'other' dataset evaluation")

    args = parser.parse_args()

    if args.evaluate == "xquad":
        evaluate_on_xquad()
    elif args.evaluate == "npho_10":
        evaluate_on_npho('10')
    elif args.evaluate == "npho_20":
        evaluate_on_npho('20')
    elif args.evaluate == "other":
        if args.file_path:
            evaluate_on_other_dataset(args.file_path)
        else:
            print("Please provide a file path for 'other' dataset evaluation.")

if __name__ == "__main__":

    os.makedirs("datasets", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    main()