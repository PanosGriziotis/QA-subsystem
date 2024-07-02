import json
import os

from haystack.nodes import FARMReader
from main import evaluate_reader, fetch_eval_dataset

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
    
    fetch_eval_dataset()
    run_readers_evaluation_test()