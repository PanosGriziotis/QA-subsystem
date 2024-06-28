> [!WARNING]
> This repository is currently a work in progress and is not yet ready for production use. Features and functionalities are being actively developed and tested. Use at your own risk.

# QA-subsystem

This repository is part of a project aiming to develop a Greek Question-Answering (QA) system that integrates with a closed-domain virtual assistant (dialogue system). As a case study, a greek-speaking conversational agent for COVID-19 is selected.

The repository contains a simple Haystack application with a REST API for indexing and querying purposes.

The application includes:

- An Elasticsearch container
- A REST API launcher built on FastAPI. This API integrates the Haystack logic and uses pipelines for indexing and querying.

You can find more information in the [Haystack documentation](https://docs.haystack.deepset.ai/v1.25/docs/intro).

## Getting started
Before you begin, make sure you have Python and Docker installed on your system. Driver Version: 470.161.03   CUDA Version: 11.4     

- Clone this repository.
- Open a terminal and run the setup.sh script. This script will create a virtual environment called venv in your working directory, where all required dependencies will be installed. Then, the Elasticsearch document store Docker container will start running. Finally, the REST API will start running in the terminal:
    ```bash
    chmod +x setup.sh && ./setup.sh
    ```
- Verify that the REST API is ready by running on a new terminal
  ```bash
  curl http://localhost:8000/ready
  ```
  You should get `true` as a response.
- Stop the REST API server by pressing `ctrl + c` in terminal.

## Indexing
To populate the application with data about covid-19, run the following script:
```bash
python ingest_dat_to_doc_store.py
```

You can also index your own text files using the `file-upload` endpoint. 

```bash
curl -X 'POST' \
'http://localhost:8000/file-upload?keep_files=false' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'files=@YOUR-TEXT-FILE'
```
Note: Acceptible file formats are .txt, .json, .jsonl, .pdf, .docx

## Querying

There are two query endpoints available for inferring answers to queries. These endpoints provide different approaches to answering queries:

### Retrieval-Augmented Generation (RAG) Query

- **Endpoint:** [http://127.0.0.1:8000/rag-query](http://127.0.0.1:8000/rag-query)
- **Description:** This endpoint utilizes a Retrieval-Augmented Generator (RAG) pipeline. It employs a domain-adapted Dense Retriever based on sentence transformers for retrieving relevant documents. The Generator is based on [Meltemi-7B-Instruct-v1](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1), an instruct version of Meltemi-7B, the first Greek Large Language Model (LLM).

### Extractive Question Answering (QA) Query

- **Endpoint:** [http://127.0.0.1:8000/extractive-query](http://127.0.0.1:8000/extractive-query)
- **Description:** This endpoint utilizes an Extractive QA pipeline based on the Retriever-Reader framework. The answer is extracted as a span from the top-ranked retrieved document. The same retrieval process is employed here. The Reader component is a fine-tuned [multilingual DeBERTaV3](https://huggingface.co/microsoft/mdeberta-v3-base) on SQuAD with further fine-tuning on COVID-QA-el_small, which is a translated small version of the COVID-QA dataset.

### Querying the application
You can query the endpoint using curl to get the full result response.

For example, to query the application with the question "Τι είναι ο covid-19;", run:
```bash
curl -X POST http://127.0.0.1:8000/rag-query \
     -H "Content-Type: application/json" \
     -d '{"query": "Πώς μεταδίδεται η covid-19;", "params": {"Retriever": {"top_k": 5}, "Generator": {"max_new_tokens": 100}}}'
```

You should get a the full response of the pipeline containing the answer, the invocation context, retrieved documents, etc.

