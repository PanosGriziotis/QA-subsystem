
from haystack.nodes.base import BaseComponent
from haystack.nodes import TransformersDocumentClassifier

from transformers import AutoTokenizer
from typing import List
import logging
from collections import Counter
from tqdm import tqdm
from haystack.document_stores import ElasticsearchDocumentStore

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)

def classify_documents (documents):
    "input haystack documents and output documents with topic label in their metadata field"
    document_classifier = TransformersDocumentClassifier(model_name_or_path="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                                                     labels =["Κοινωνία & Πολιτισμός", "Επιστήμη & Μαθηματικά", "Υγεία", "Εκπαίδευση", "Υπολογιστές & Διαδίκτυο", "Αθλητισμός", "Επιχειρήσεις & Οικονομικά", "Ψυχαγωγία & Μουσική", "Οικογένεια & Σχέσεις", "Πολιτική & Κυβέρνηση"],
                                                     task = 'zero-shot-classification')        

    return document_classifier.predict_batch(documents=documents)

def get_label_stats (labeled_documents):
    "get statistics on label prevelence for a list of documents"

    labels = [doc.to_dict()["meta"]["classification"]["label"] for doc in labeled_documents]
    label_counts = Counter(labels)
    total_labels = len(labels)
    label_percentages = {label: count / total_labels for label, count in label_counts.items()}
    
    return label_percentages

def index_in_ds (o_index, document_store, documents):
        document_store.delete_documents(index=o_index)
        document_store.write_documents(documents, index=o_index, duplicate_documents="overwrite")

def main (index="filtered_wiki", document_store=ElasticsearchDocumentStore(), o_index = None):

    docs = document_store.get_all_documents(index=index)
    
    labeled_docs = classify_documents(docs)
    print (get_label_stats(labeled_docs))
    
    if o_index is not None:
         index_in_ds(o_index=o_index, document_store=document_store, documents=labeled_docs)

    return labeled_docs

if __name__ == '__main__':
    main()
"""
class DomainFilter(BaseComponent):
      # If it's not a decision component, there is only one outgoing edge
    outgoing_edges = 1
    def __init__(self):
        super().__init__()

    def run(self, documents:list,  exclusion_label="Άλλο", **kwargs):
    # Insert code here to manipulate the input and produce an output dictionary
    
        filtered_docs = []
        for doc in tqdm(documents):
                # get labels with scores
                label = doc['meta']['classification']['label']
                if label != exclusion_label:
                    filtered_docs.append(doc)        
                else: 
                    continue
        output = { "documents": filtered_docs}
        
        

        logging.info(f"Documents filtered. Num of doc before & after filtering: {len(documents)}\t{len(filtered_docs)}")
        return output, 'output_1'

    def run_batch(
        self,
        **kwargs):
         return

"""