from haystack.nodes import BM25Retriever, EmbeddingRetriever, DensePassageRetriever, SentenceTransformersRanker
from haystack.document_stores import ElasticsearchDocumentStore
from evaluate import evaluate_retriever, evaluate_reader
import json
from haystack.nodes import FARMReader

from datasets import load_dataset

# ==================== LOAD & SAVE NPHO DATASET =========================================

# Load the annotated test dataset
npho = load_dataset("panosgriz/npho-covid-SQuAD-el")

# Extract specific entries from the dataset 
npho_10 = npho["test"][0]
npho_20 = npho["test"][1]

# Save the extracted entries to JSON files
datasets = {'10': npho_10, '20': npho_20}
for key, dataset in datasets.items():
    with open(f"./test_npho_{key}.json", "w") as fp:
        json.dump(dataset, fp, ensure_ascii=False, indent=6)

#=========================== EVALUATE RETRIEVERS ======================================================


ds = ElasticsearchDocumentStore()

emb_retriever = EmbeddingRetriever(document_store=ds, embedding_model="panosgriz/covid_el_embedding_retriever", max_seq_len=128)
dpr = DensePassageRetriever(
    document_store=ds,  
    query_embedding_model="panosgriz/bert-base-greek-covideldpr-query_encoder",
    passage_embedding_model="panosgriz/bert-base-greek-covideldpr-ctx_encoder",
    max_seq_len_passage=356)
bm25 = BM25Retriever(document_store=ds)

top_k_values = [x for x in range(1,51)]


results = {}
for name, retriever in {"emb_retriever":emb_retriever, "dpr":dpr, "bm25":bm25}.items():
    reports = evaluate_retriever(retriever=retriever, document_store=ds, eval_filename="test_npho20.json", top_k_list=top_k_values)
    results[name] = reports

with open ("./retrievers_eval_report_npho20.json", "w") as fp:
    json.dump (results, fp, ensure_ascii=False, indent=4)


# ======================================= EVALUATE READERS ===============================================

"""

# List of reader models
readers = [
    "panosgriz/xlm-roberta-squad2-covid-el_small",
    "panosgriz/xlm-roberta-squad2-covid_el",
    "panosgriz/mdeberta-v3-base-squad2-covid-el",
    "panosgriz/mdeberta-v3-base-squad2-covid-el_small",
    "panosgriz/xlm-roberta-covid-small-early-stopping",
    "system/experiments/readers/mdeberta-v3-base-squad2_covid_el_small"
]

# Initialize FARMReader objects
readers = [FARMReader(model_name_or_path=reader) for reader in readers]

# Evaluate readers on the test data and save the reports
for datafile in ["test_npho10.json", "test_npho20.json"]:
    reports = {}
    for reader in readers:
        reports[reader.model_name_or_path] = evaluate_reader(reader, datafile)
    with open(f"system/evaluation/report_readers_{os.path.basename(datafile).split('.')[0]}.json", "w") as fp:
        json.dump(reports, fp, ensure_ascii=False, indent=4)

"""
"""
def evaluate_reader_on_top_k (model_name_or_path:str ="panosgriz/mdeberta-v3-base-squad2-covid-el_small",
                              top_k_values:list = [x for x in range(0,51)],
                              datafile:str = "eval_data/test_npho20.json"):
    
    reader = FARMReader(model_name_or_path=model_name_or_path)

    top_k_v = [x for x in range(0,51)]

    for datafile in ["test_npho10.json", "test_npho20.json"]:

        reports = evaluate_reader(reader = reader, eval_filename=datafile, top_k_list=top_k_v)

        with open(f"system/evaluation/report_mdeberta_small_reader_top_k_{os.path.basename(datafile).split('.')[0]}.json", "w") as fp:
            json.dump(reports, fp, ensure_ascii=False, indent=4)
"""