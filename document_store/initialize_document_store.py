import os
import time
import requests
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import launch_es


def check_elasticsearch():
    """Check if Elasticsearch is up and running."""
    url = f"http://localhost:9200/_cat/health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

def initialize_document_store():
    """Initialize a Elasticsearch document store object."""
    return ElasticsearchDocumentStore(
        host="localhost",
        port=9200,
        username="",
        password="",
        index="test_api",
        embedding_dim=768,
        duplicate_documents="overwrite"
    )

document_store = None
if check_elasticsearch():
    document_store = initialize_document_store()
    
if __name__ == "__main__":
    # Start Elasticsearch docker container
    launch_es()

    # Wait for Elasticsearch to be ready
    print("Checking for active Elasticsearch service...")
    retries = 10
    while retries > 0:
        if check_elasticsearch():
            print("Elasticsearch is up and running.")
            break
        else:
            print("Waiting for Elasticsearch to start...")
            retries -= 1
            time.sleep(5)
    else:
        print("Elasticsearch did not start in time.")
        exit(1)

    # Initialize the document store
    es_ds = initialize_document_store()
    print("Elasticsearch document store initialized.")
