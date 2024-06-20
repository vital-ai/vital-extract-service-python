import uvicorn
import sys
from fastapi import FastAPI, HTTPException
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from pydantic import BaseModel
from typing import Any, Dict
from extractservice.extract_service import ExtractService
from extractservice.utils.config_utils import ConfigUtils

app = FastAPI()

logger = logging.getLogger("ExtractServiceLogger")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = jsonlogger.JsonFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

config = ConfigUtils.load_config()

port = config['extract_service']['port']

extract_service = ExtractService()


@app.get("/health")
async def health_check():
    logger.info("health check")
    return {"status": "ok"}


class ExtractRequest(BaseModel):
    task: Dict[str, Any]
    data: Dict[str, Any]


class ExtractResult(BaseModel):
    results: Dict[str, Any]
    status: str


@app.post("/extract", response_model=ExtractResult)
async def extract_data(request: ExtractRequest):
    try:
        task = request.task
        data = request.data

        doc_id = data['document_list'][0]['doc_id']
        doc_content = data['document_list'][0]['doc_content']

        logger.info({"doc_id": doc_id, "doc_content": doc_content})

        extract_service.extract(doc_id, doc_content)

        processed_results = {"key": "value"}

        logger.info(processed_results)

        result = ExtractResult(results=processed_results, status="ok")

        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=port, app=app)
