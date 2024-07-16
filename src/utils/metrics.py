
import os 
import sys
import json
from haystack.schema import Document 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import bisect
from sentence_transformers import SentenceTransformer

def compute_answer_relevancy (query, answers):
    
    model = SentenceTransformer("panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2")
    
    a_embeddings = model.encode(sentences=answers)
    q_embedding = model.encode(sentences=[query])

    # Get the similarity scores for the embeddings (tensor). Matrix of [top_k, 1] size
    similarities = model.similarity(a_embeddings, q_embedding)

    flat_similarities = [item[0] for item in similarities.tolist()]

    return flat_similarities

def add_relevancy_scores_to_results (results):
    
    query = results["query"]
    answers = results["answers"]
    answer_texts = [answer.answer for answer in answers]
    scores = compute_answer_relevancy(query=query, answers=answer_texts)
    
    scored_answers = []
    for score, answer in zip(scores, answers):
        answer.meta["relevancy_score"] = score
        # Use bisect to find the position where the current answer should be inserted
        insert_position = bisect.bisect_right([a.meta["relevancy_score"] for a in scored_answers], score)
        # Insert the current answer into the correct position to maintain sorted order
        scored_answers.insert(len(scored_answers) - insert_position, answer)

    results ["answers"] = scored_answers
    
    return results