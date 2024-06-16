from datasets import load_dataset
import json
from evaluate import evaluate_reader
from haystack.nodes import FARMReader
import logging
import os 
"""
# Load the dataset
npho = load_dataset("panosgriz/npho-covid-SQuAD-el")

# Extract specific entries from the dataset
npho_10 = npho["test"][0]
npho_20 = npho["test"][1]

# Save the extracted entries to JSON files
datasets = {'10': npho_10, '20': npho_20}
for key, dataset in datasets.items():
    with open(f"test_npho{key}.json", "w") as fp:
        json.dump(dataset, fp, ensure_ascii=False, indent=6)

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
reader = FARMReader(model_name_or_path="panosgriz/mdeberta-v3-base-squad2-covid-el_small")

top_k_v = [x for x in range(0,51)]

for datafile in ["test_npho10.json", "test_npho20.json"]:

    reports = evaluate_reader(reader = reader, eval_filename=datafile, top_k_list=top_k_v)

    with open(f"system/evaluation/report_mdeberta_small_reader_top_k_{os.path.basename(datafile).split('.')[0]}.json", "w") as fp:
        json.dump(reports, fp, ensure_ascii=False, indent=4)