import glob
import requests
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_status_endpoint():

    r = requests.get(url="http://127.0.0.1:8000/ready")    
    assert r.status_code == 200
    assert r.text == "true"
    logging.info("Status endpoint test passed.")

def test_file_upload_endpoint():

    logging.info("Testing the file upload endpoint...")
    start_time = time.time()

    for txt_file in glob.glob("example_data/*.txt"):
        with open(txt_file, "rb") as f:
            logging.info(f"Uploading file: {txt_file}")
            r = requests.post(url="http://127.0.0.1:8000/file-upload", files={"files": f})
            json_response = r.json()
            assert len(json_response["documents"]) > 0
    
    end_time = time.time()
    logging.info(f"File upload endpoint test passed. (Time taken: {end_time - start_time:.4f} seconds)")

def test_rag_query_endpoint():

    logging.info("Testing the RAG query endpoint...")
    query = "Πώς μεταδίδεται ο covid-19;"
    logging.info(f"Query: {query}")
    start_time = time.time()

    r = requests.get(url="http://127.0.0.1:8000/rag-query", params={"query": query})
    json_response = r.json()
    logging.info(f"Received response: {json_response}")
    
    assert json_response['answers']
    assert len(json_response['documents']) > 0
    assert json_response['query'] == query
    end_time = time.time()
    
    logging.info(f"RAG query endpoint test passed. (Time taken: {end_time - start_time:.4f} seconds)")

def test_extractive_query_endpoint():
    
    logging.info("Testing the extractive query endpoint...")
    query = "Πώς μεταδίδεται ο covid-19;"
    logging.info(f"Query: {query}")
    start_time = time.time()
    
    r = requests.get(url="http://127.0.0.1:8000/extractive-query", params={"query": query})
    json_response = r.json()
    logging.info(f"Received response: {json_response}")
    
    assert json_response['answers']
    assert len(json_response['documents']) > 0
    assert json_response['query'] == query
    
    end_time = time.time()
    logging.info(f"Extractive query endpoint test passed. (Time taken: {end_time - start_time:.4f} seconds)")

if __name__ == "__main__":

    test_status_endpoint()
    test_file_upload_endpoint()
    test_extractive_query_endpoint()
    test_rag_query_endpoint()

    logging.info(f"All tests completed successfully")
