"""
Pipelines allow putting together Components to build a graph.

In addition to the standard Haystack Components, custom user-defined Components
can be used in a Pipeline YAML configuration.

The classes for the Custom Components must be defined in this file.
"""

import os
import sys 

from haystack.nodes.base import BaseComponent

from generator import build_generator

class Generator(BaseComponent):
    outgoing_edges = 1

    def __init__(self, prompt_messages=None, max_new_tokens=None, temperature=None, top_p=None):
        
        self.generator = build_generator(messages=prompt_messages, 
                                         max_new_tokens=max_new_tokens, 
                                         temperature=temperature, 
                                         top_p=top_p)
        super().__init__()

    def run(self, query, documents):

        result, _ = self.generator.run(query=query, documents=documents)
        answer = result['answers'][0].answer
        output = {"generator_answer": answer}
        return output, 'output_1'
    
    def run_batch(
        self):
         return