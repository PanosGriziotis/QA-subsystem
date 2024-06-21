from typing import List, Optional
from pathlib import Path
import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
import logging

from pipelines.rag_pipeline import rag_pipeline
from pipelines.extractive_qa_pipeline import extractive_qa_pipeline
from pipelines.indexing_pipeline import indexing_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QA-subystem API")

# Create the file upload directory if it doesn't exist
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent.parent / "file-upload").absolute()))
Path(FILE_UPLOAD_PATH).mkdir(parents=True, exist_ok=True)


@app.get("/ready")
def check_status():
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@app.post("/file-upload")
def upload_files(files: List[UploadFile] = File(...), keep_files: Optional[bool] = False, recreate_index: Optional[bool]=False):
    """
    Upload a list of files to be indexed.

    Note: files are removed immediately after being indexed. If you want to keep them, pass the
    `keep_files=true` parameter in the request payload.
    """

    file_paths: list = []

    for file_to_upload in files:
        try:
            
            file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file_to_upload.filename}"
            
            with file_path.open("wb") as fo:
                
                fo.write(file_to_upload.file.read())
            
            file_paths.append(file_path)
        
        finally:
            
            file_to_upload.file.close()

    result = indexing_pipeline(recreate_index=recreate_index).run(file_paths=file_paths)

    # Clean up indexed files
    if not keep_files:
        for p in file_paths:
            p.unlink()

    return result


# Extractive QA pipeline endpoint
@app.get("/extractive-query")
def ask_retriever_reader_pipeline(query: str):
    
    try:
        result = extractive_qa_pipeline().run(query=query)
        
        return result
    
    except Exception as e:
        
        logger.error(e)
        
        raise HTTPException(status_code=500, detail="Running Pipeline Error")

@app.get("/rag-query")
def ask_rag_pipeline(query: str):
    try:
        result = rag_pipeline().run(query=query)
        
        return result
    
    except Exception as e:
        
        logger.error(e)
        
        raise HTTPException(status_code=500, detail="Running Pipeline Error")

