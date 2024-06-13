from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from typing import List, Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from haystack.nodes.base import BaseComponent

import logging
import argparse

import json 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordFilterer(BaseComponent):
    """Filter component to be used as a node in a haystack indexing pipeline"""

    outgoing_edges = 1
    def __init__(self):
        super().__init__()

    def run(self, documents:list, keywords: list = ["πανδημία covid-19"], top_k = 200, **kwargs):

        document_store = InMemoryDocumentStore(use_bm25=True)
        document_store.write_documents(documents)
        filtered_docs = filter_documents_from_index (keywords = keywords, top_k = top_k, input_index = document_store.index, document_store=document_store)
        output = { "documents": filtered_docs}     

        logging.info(f"Documents filtered with keyword matching. Num of docs before: {len(documents)}\tNum of docs after: {len(filtered_docs)}")
        
        return output, 'output_1'

    def run_batch(
        self,
        **kwargs):
         return

def filter_documents_from_index (keywords: List[str], top_k: int, input_index: str, document_store: Union[ElasticsearchDocumentStore,InMemoryDocumentStore] = ElasticsearchDocumentStore(), output_index: Optional[str] = None) -> None:
    """Filter documents in a document store using BM25 retriever to keep only documents related to a given list of keywords."""

    retriever = BM25Retriever(document_store)

    # inspect input index

    logging.info(f"{document_store.get_document_count(index=input_index)} documents in index {input_index}")
    
    document_store.get_document_count(index=input_index)

    if len(keywords) > 1:
        lists_of_filtered_docs = retriever.retrieve_batch(queries=keywords,  top_k=top_k, index=input_index, scale_score = True)
        filtered_docs=[]
        for keyword_top_docs in lists_of_filtered_docs:
            for doc in keyword_top_docs:
                filtered_docs.append(doc)
        filtered_docs = remove_duplicates(filtered_docs)
        
    else: 
        filtered_docs = retriever.retrieve(keywords[0], top_k=top_k, index=input_index, scale_score = True)
    
    if output_index is not None:
        document_store.delete_documents(index=output_index)
        document_store.write_documents(filtered_docs, index=output_index, duplicate_documents="overwrite")
        
    else: 
        return filtered_docs
        
def remove_duplicates(docs):
    doc_ids = set()
    unique_docs = []
    for doc in docs:
        if doc.id not in doc_ids:
            unique_docs.append(doc)
            doc_ids.add(doc.id)
    return unique_docs

def calculate_mean_score(docs: List[Dict[str, Any]]) -> float:
    """Calculate the mean score of the retrieved documents."""
    
    scores = [doc.score for doc in docs]
    return np.mean(scores)

def plot_mean_scores(results: Dict[int, float], save_path: str = "./top_k_mean_scores_plot.png") -> None:
    """Plot the mean scores vs top_k values."""
    
    top_k_values = list(results.keys())
    mean_scores = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(top_k_values, mean_scores, marker='o')
    plt.xlabel('Top K Values')
    plt.ylabel('Mean Score of Retrieved Documents')
    plt.title('Mean Score vs Top K Values')
    plt.grid(True)
    
    logger.info(f"Saving plot to {save_path}")
    try:
        plt.savefig(save_path)  # Save the plot as an image
        logger.info(f"Plot successfully saved as {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    plt.show()


def experiment(top_k_values: List[int], keywords: List[str], input_index: str,  document_store: Union[ElasticsearchDocumentStore,InMemoryDocumentStore]=ElasticsearchDocumentStore()) -> None:
    """Experiment with different top_k values and plot the mean scores."""
    
    results = {}

    for top_k in top_k_values:

        logger.info(f"Starting experiment with top_k={top_k}")
        
        # Filter documents for the current top_k value
        filtered_docs = filter_documents_from_index(keywords, top_k, input_index, document_store)

        logger.info(f"Retrieved {len(filtered_docs)} unique documents for top_k={top_k}")
        
        # Calculate the mean score for the current top_k
        mean_score = calculate_mean_score(filtered_docs)
        if top_k > 0 and round(mean_score, 2) == 0.85:
            logging.info (f"cutoff 0.85 reached at top_k = {top_k}")
        results[top_k] = mean_score

    # Plot the results
    plot_mean_scores(results)

def rank_docs (keywords, documents, top_k):
    ranker = SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco",
                                        top_k=top_k
                                        )
    ranked_docs = ranker.predict(query=keywords[0],documents=documents)
    return ranked_docs
    #ranked_docs = ranker.run_batch(queries=keywords, documents=documents )
    

    
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default="wiki", help="index name in document store for fetching the documents")
    parser.add_argument('--keywords',  nargs='+', default=["πανδημία covid-19"], help='list of queries for retrieving relevant documents')
    parser.add_argument('--o_index', type=str, default=None, help="index for storing the filtered documents")
    parser.add_argument('--top_k', type=str, default=650,  help="number of documents to keep after retrieving and ranking")
    parser.add_argument('--experiment', action='store_true', help="experiment with different top_k values" )
    args = parser.parse_args()

    if args.experiment:
        top_k_values = [x for x in range(0, 1001, 50)]
        experiment(top_k_values=top_k_values, keywords=args.keywords, input_index=args.index)
    
    filtrered_docs = filter_documents_from_index(keywords=args.keywords, top_k=args.top_k, input_index=args.index, output_index=args.o_index)
    
    ## EXTRA CODE ##
    def rank_docs (keywords, documents, top_k):
        ranker = SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco",
                                            top_k=top_k
                                            )
        ranked_docs = ranker.predict(query=keywords[0],documents=documents)
        return ranked_docs
    #ranked_docs = ranker.run_batch(queries=keywords, documents=documents )
    
    
    docs = rank_docs(keywords=args.keywords, documents=filtrered_docs, top_k=args.top_k)

    with open (f"filtered_docs_{args.top_k}", "w") as fp:
        for doc in docs:
            json_line = json.dumps(doc.to_dict(), ensure_ascii=False)
            fp.write(json_line + "/n" )
