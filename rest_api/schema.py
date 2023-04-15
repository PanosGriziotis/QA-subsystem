from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, Extra
from pydantic import BaseConfig
from haystack import Answer, Document
PrimitiveType = Union[str, int, float, bool]

class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid

class QueryRequest(RequestBaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False

class FilterRequest(RequestBaseModel):
    filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None

class QueryResponse(BaseModel):
    query: str
    answers: List[Answer] = []
    documents: List[Document] = []
    debug: Optional[Dict] = Field(None, alias="_debug")