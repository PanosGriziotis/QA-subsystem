from haystack.pipelines import Pipeline

query_pipeline = Pipeline.load_from_yaml(path="rag.haystack-pipeline.yml", pipeline_name="query")

results = query_pipeline.run(query="Πόσες μέρες πρέπει να μείνω σε καραντίνα;")

print (results)