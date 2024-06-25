import json
from typing import List

from haystack.schema import Document

from deepeval.metrics.ragas import RAGASAnswerRelevancyMetric
from deepeval.metrics.ragas import RAGASFaithfulnessMetric
from deepeval.metrics import FaithfulnessMetric

from deepeval.test_case import LLMTestCase
from deepeval import evaluate


from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from datasets import load_dataset

from system.rest_api.rest_api.pipeline.rag_pipeline import build_generator
import os
# 1. Evaluate generator on given ground truth context.
# Load the dataset
os.environ['OPENAI_API_KEY'] = "sk-thesis-service-account-lSBHDMCXJ8WcpY4uhvXRT3BlbkFJUDJgyhJgvjhDRAY4ROWU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
def create_goldens_from_squad (squad_data):
    """Create evaluation dataset from SQuAD-like dataset"""

    examples = []

    #for article in squad_data["data"]:

    for paragraph in squad_data["data"][0]["paragraphs"]:

            context = paragraph["context"]

            for qas in paragraph["qas"][:1]:

                golden  = Golden(input=qas["question"], context=[context])
                golden.expected_output = qas["answers"][0]['text']            
                
                examples.append(golden)
                
    return examples


def convert_goldens_to_test_cases(goldens: List[Golden], generator):
    """"""

    test_cases = []

    for golden in goldens:
        
        query = golden.input
        context = Document(content = golden.context[0])
        result, _ = generator.run (query = query, documents = [context])
        actual_output = result["answers"][0].answer

        test_case = LLMTestCase(
            input=query,
            actual_output= actual_output,
            expected_output= golden.expected_output,
            retrieval_context = golden.context
        )
        test_cases.append(test_case)

    return test_cases

# Data preprocessing before setting the dataset test cases
#dataset.test_cases = convert_goldens_to_test_cases(dataset.goldens)

if __name__ == "__main__":

    npho = load_dataset("panosgriz/npho-covid-SQuAD-el")

    goldens = create_goldens_from_squad(npho["test"][1])

    messages = [
        {"role": "system", "content": "Χρησιμοποιώντας τις πληροφορίες που περιέχονται στο Κείμενο, δώσε μια ολοκληρωμένη απάντηση στην Ερώτηση. Εάν η απάντηση δεν μπορεί να συναχθεί από το Κείμενο, απάντα 'Δεν ξέρω'."},
        {"role": "user", "content": "Ερώτηση: {query}; Κείμενο: {to_strings(documents)[0]}"}
    ]
    
    generator = build_generator(messages)

    test_cases = convert_goldens_to_test_cases(goldens, generator)

    dataset = EvaluationDataset(test_cases)
    
    result = dataset.evaluate([RAGASAnswerRelevancyMetric(model= "gpt-3.5-turbo")])
    print ("RESULT:", result)
    



"""

# Replace this with the actual output from your LLM application
actual_output = ""

# Replace this with the expected output from your RAG generator
expected_output = ""

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = [""]

metric = RAGASFaithfulnessMetric(threshold=0.5, model="gpt-3.5-turbo")

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

metric.measure(test_case)
print(metric.score)

# or evaluate test cases in bulk
evaluate([test_case], [metric])
"""