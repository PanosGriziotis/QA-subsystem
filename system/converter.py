from transformers import PreTrainedTokenizer, BatchEncoding, MT5Tokenizer
from typing import Optional, List
from haystack import Document

class _MT5inputConverter:
    def __call__(
        self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> BatchEncoding:
        conditioned_doc = " " + " ".join([d.content for d in documents])

        # concatenate question and support document into MT5 input
        query_and_docs = " {}<\s>{}<\s>".format(query, conditioned_doc)

        return tokenizer([(query_and_docs)], truncation=True, padding=True, return_tensors="pt")

