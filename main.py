"""
SumerTrip AI Services API
Chatbot + RAG Recommendation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

from config import config
from chatbot import get_chat_response
from recommendation import get_recommendations, update_data


# Models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    temperature: float = Field(0.7, ge=0, le=2)


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str]
    usage: Dict[str, Any]


class RecommendRequest(BaseModel):
    query: str = Field(..., description="User info/preferences as string")
    top_k: int = Field(5, ge=1, le=20)
    item_type: Optional[str] = Field(None, description="'trip' or 'event'")


class RecommendItem(BaseModel):
    id: Optional[int]
    type: str
    score: float
    data: Dict[str, Any]


class DataUpdateRequest(BaseModel):
    trips: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []


# App
app = FastAPI(title="SumerTrip AI Services", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
def health():
    return {"status": "healthy", "services": ["chatbot", "recommendation"]}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chatbot for Iraqi tourism questions"""
    try:
        return await get_chat_response(
            message=request.message,
            conversation_id=request.conversation_id,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/recommend", response_model=List[RecommendItem])
def recommend(request: RecommendRequest):
    """
    Get recommendations based on user query string.

    Example: "أحب التاريخ والآثار، ميزانية متوسطة"
    """
    try:
        return get_recommendations(
            query=request.query, top_k=request.top_k, item_type=request.item_type
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/data/update")
def update_rag(request: DataUpdateRequest):
    """Update events/trips in RAG"""
    try:
        count = update_data(request.trips, request.events)
        return {"status": "ok", "items": count}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
