
from fastapi import FastAPI, File, UploadFile
import subprocess
import uvicorn 
from haystack.pipelines.base import Pipeline
from pathlib import Path
import logging
from schema import RequestBaseModel, QueryResponse, QueryRequest
from system.pipelines import extractive_qa_pipeline, indexing_pipeline
from typing import Dict, Any, List
import time 
import logging
import json
import sys

logger = logging.getLogger("haystack")

app = FastAPI(title="Haystack REST API", debug=True)
query_pipeline = extractive_qa_pipeline()


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def process_request(request:QueryRequest) -> Dict[str, Any]:
    
    start_time = time.time()

    params = request.params or {}

    result = query_pipeline.run(query=request.query, params=params, debug=request.debug)

    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in result:
        result["documents"] = []
    if not "answers" in result:
        result["answers"] = []

    logger.info(
        json.dumps({"request": request, "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str)
    )
    return result

if __name__ == "__main__":
  #file_path = sys.argv[1]
  indexing_pipeline().run(file_paths = ['data/eody/covid_data.txt'])
  uvicorn.run("app:app", port=8000, reload=True, access_log=False)




