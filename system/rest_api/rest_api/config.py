import os
from pathlib import Path


PIPELINE_CONFIG_PATH = os.getenv(
    "PIPELINE_CONFIG_PATH", str((Path(__file__).parent / "pipeline" / "pipelines.config.yml").absolute())
)
QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "extractive_query_pipeline")
INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent / "file-upload").absolute()))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")

CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", "4"))
