from fastapi import FastAPI
from pydantic import BaseModel, Field

from llm import get_recommendation


class HistoryItem(BaseModel):
    tmdb_id: int
    name: str


class RecommendRequest(BaseModel):
    user_id: int
    preferences: str
    history: list[HistoryItem] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    user_id: int
    tmdb_id: int
    description: str


app = FastAPI(title="Agentic Movie Recommender")


@app.get("/")
def read_root() -> dict:
    return {"status": "ok", "message": "Movie recommender is running."}


@app.get("/kaithhealth")
def kaithhealth() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    history_names = [item.name for item in request.history]
    history_ids = [item.tmdb_id for item in request.history]

    recommendation = get_recommendation(
        request.preferences,
        history_names,
        history_ids,
    )

    return RecommendResponse(
        user_id=request.user_id,
        tmdb_id=int(recommendation["tmdb_id"]),
        description=str(recommendation["description"]),
    )
