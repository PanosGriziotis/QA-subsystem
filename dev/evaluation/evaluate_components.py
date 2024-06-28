from haystack.nodes import BM25Retriever, EmbeddingRetriever, DensePassageRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from main import evaluate_retriever, evaluate_reader, fetch_eval_dataset
import json
from haystack.nodes import FARMReader
import os

DOCUMENT_STORE = ElasticsearchDocumentStore()

def run_retriever_evaluation_test ():
 
    emb_retriever = EmbeddingRetriever(document_store=DOCUMENT_STORE, embedding_model="panosgriz/covid_el_embedding_retriever", max_seq_len=128)
    dpr = DensePassageRetriever(
        document_store=DOCUMENT_STORE,  
        query_embedding_model="panosgriz/bert-base-greek-covideldpr-query_encoder",
        passage_embedding_model="panosgriz/bert-base-greek-covideldpr-ctx_encoder",
        max_seq_len_passage=356)
    bm25 = BM25Retriever(document_store=DOCUMENT_STORE)

    top_k_values = [x for x in range(1,51)]

    results = {}
    for name, retriever in {"bm25":bm25, "embedding_retriever": emb_retriever, "dpr": dpr}.items():
        reports = evaluate_retriever(retriever=retriever, document_store=DOCUMENT_STORE, eval_filename="test_npho_20.json", top_k_list=top_k_values)
        results[name] = reports

    with open ("reports/retrievers_eval_report_npho20.json", "w") as fp:
        json.dump (results, fp, ensure_ascii=False, indent=4)

def run_readers_evaluation_test():
    # List of reader models
    readers = [
        "panosgriz/xlm-roberta-squad2-covid-el_small",
        "panosgriz/xlm-roberta-squad2-covid_el",
        "panosgriz/mdeberta-v3-base-squad2-covid-el",
        "panosgriz/mdeberta-v3-base-squad2-covid-el_small",
        "panosgriz/xlm-roberta-covid-small-early-stopping",
    ]

    # Initialize FARMReader objects
    readers = [FARMReader(model_name_or_path=reader) for reader in readers]

    # Evaluate readers on the test data and save the reports
    for datafile in ["test_npho_10.json", "test_npho_20.json"]:
        reports = {}
        for reader in readers:
            reports[reader.model_name_or_path] = evaluate_reader(reader, datafile)
        with open(f"reports/report_readers_{os.path.basename(datafile).split('.')[0]}.json", "w") as fp:
            json.dump(reports, fp, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    report_dir = "./reports"
    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)
        
    #fetch_eval_dataset()
    #run_retriever_evaluation_test()
    run_readers_evaluation_test()