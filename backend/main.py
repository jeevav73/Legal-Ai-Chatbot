import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import rag_engine
import complaint_generator

app = FastAPI(title="RightsGuard Backend")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    question: str


class ComplaintDetails(BaseModel):
    name: str
    address: str
    contact: str
    addressee: str
    incident_date: str
    location: str
    accused: str
    witnesses: str
    evidence: str
    relief_requested: str


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        result = rag_engine.answer_question(request.question)
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as exc:
        detail = str(exc)
        raise HTTPException(status_code=500, detail=detail)


@app.post("/api/complaint")
def complaint_endpoint(details: ComplaintDetails):
    try:
        file_path = complaint_generator.generate_complaint(details.dict())
        return FileResponse(
            path=file_path,
            filename=Path(file_path).name,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception as exc:
        detail = str(exc)
        raise HTTPException(status_code=500, detail=detail)


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
