from haystack.nodes import BM25Retriever, EmbeddingRetriever, DensePassageRetriever, SentenceTransformersRanker
from haystack.document_stores import ElasticsearchDocumentStore
from evaluate import evaluate_retriever
import json

max_seq_len = 256

ds = ElasticsearchDocumentStore()

emb_retriever = EmbeddingRetriever(document_store=ds, embedding_model="system/experiments/retrievers/adapted_embedding_retriever/adapted_retriever_expanded", max_seq_len=max_seq_len)

dpr = DensePassageRetriever(
    document_store=ds,  
    query_embedding_model="/home/pgriziotis/thesis/QA-subsystem/system/experiments/retrievers/dpr/dpr_trained_new/query_encoder",
    passage_embedding_model="/home/pgriziotis/thesis/QA-subsystem/system/experiments/retrievers/dpr/dpr_trained_new/passage_encoder",
    max_seq_len_passage=max_seq_len)

bm25 = BM25Retriever(document_store=ds)

top_k_values = [x for x in range(1,51)]

"""
ranker = SentenceTransformersRanker(
    model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco"
)
"""
results = {}
for name, retriever in {"emb_retriever":emb_retriever, "dpr":dpr}.items():
    reports = evaluate_retriever(retriever=retriever, document_store=ds, eval_filename="test_npho20.json", top_k_list=top_k_values)
    results[name] = reports

with open ("system/evaluation/retrievers_eval_report_npho20.json", "w") as fp:
    json.dump (results, fp, ensure_ascii=False, indent=4)