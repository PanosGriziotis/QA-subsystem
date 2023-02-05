# QA-subsystem-for-Theano

This is a project aiming to develop a Question Answering (QA) system that integrates with a virtual assistant (dialogue system). As a case study THEANO is selected, a greek-speaking conversational agent for COVID-19 built with the open-source framework of Rasa. 

# Installing dependencies

## Quick Install

* Working on python == 3.8.10    

* create a virtual environment.
```
 python3 -m venv <virtual-environment-name>
```

* activate the virtual environment.

```
source <virtual-environment-name>/bin/activate
```

* update pip 

```
pip install --upgrade pip
```

* Install the required python packages

```
pip3 install -r requirements.txt
```
## Initialize the system and ask questions locally

* To initialize an ExtractiveQA pipeline instance and get an answer to a question right away run the following command:

```
python3 system/pipeline.py "your question"
```

Make sure your question is in greek language and related to covid-19 (e.g. "Τι θα πάθω αν κολλήσω covid ενώ είμαι εγκυος;")

## Create an HTTP API that runs the system

### 1. Create a quick FAST API
### 2. Create a Haystack REST API 


## Ask the system


