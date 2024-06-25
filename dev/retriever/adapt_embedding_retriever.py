
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore  # keep it here !
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes.retriever.sparse import BM25Retriever  # keep it here !  # pylint: disable=unused-import
from haystack.nodes.label_generator import PseudoLabelGenerator
import json
from haystack.nodes import PreProcessor
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


# GLOBAL VARIABLES


# STEP 1: GET INPUT DATA

# merge


# STEP 2: WRITE INPUT DATA INTO DOCUMENT STORE
corpus = []
with open ("./gpl_data.json", "r") as fp:
    qd_pairs = json.load(fp)
    
for qd_pair in qd_pairs:
    corpus.append (qd_pair["document"])

document_store = ElasticsearchDocumentStore(host="localhost", port = 9200, username="", password = "", index = "gpl")

document_store.delete_documents ()

preprocessor = PreProcessor(
    clean_empty_lines=False,
    clean_whitespace=False,
    split_respect_sentence_boundary=False,
    )

docs = preprocessor.process([{"content": t} for t in corpus])

document_store.write_documents(docs)

# Run pseudo label generator to create training data for the embedding retriever. This function is a) retrieving negative passages for each query-passage pair b) calculates margin loss between positive and negative passages as a score
bm25_retriever = BM25Retriever(document_store=document_store)

cross_encoder_model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"

psg = PseudoLabelGenerator(qd_pairs, bm25_retriever, cross_encoder_model_name_or_path= cross_encoder_model_name )
output, _ = psg.run(documents = document_store.get_all_documents())

# STEP 5: Adapt the retriever based on the generated train examples.

# INITIALIZE EMBEDDING RETRIEVER
with open ("./gpl_training_data.json", "w") as fp:
    fp.write (str(output["gpl_labels"]))

bi_encoder = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
max_seq_len = SentenceTransformer(bi_encoder).max_seq_length

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_format="sentence_transformers",
    max_seq_len=max_seq_len,
    progress_bar=True,
)
document_store.update_embeddings(retriever)

# TRAIN THE RETRIEVER
retriever.train (output["gpl_labels"])
retriever.save(save_dir = "./adapted_retriever_expanded")


