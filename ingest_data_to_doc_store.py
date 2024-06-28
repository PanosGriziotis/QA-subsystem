import glob
import requests
import logging
logging.basicConfig(level=logging.INFO)

def ingest_index_data():
    """
    Call the file-upload endpoint with all the files in the index data folder.
    """ 
    for file in glob.glob("index_data/*"):
        logging.info(f"Indexing content in {file} to document store")
        with open(file, "rb") as f:
            requests.post(url="http://127.0.0.1:8000/file-upload", files={"files": f})   

if __name__ == "__main__":
    ingest_index_data()
