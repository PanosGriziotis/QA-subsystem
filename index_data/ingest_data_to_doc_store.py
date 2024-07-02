import glob
import requests
import logging
import os

logging.basicConfig(level=logging.INFO)

def ingest_index_data():
    """
    Call the file-upload endpoint with all the files in the index data folder.
    """ 
    
    dir = os.path.dirname(os.path.abspath(__file__))
    for file in glob.glob(f"{dir}/*.jsonl"):
        logging.info(f"Indexing content in {file} to document store")
        with open(file, "rb") as f:
            requests.post(url="http://127.0.0.1:8000/file-upload", files={"files": f})   

if __name__ == "__main__":
    ingest_index_data()
