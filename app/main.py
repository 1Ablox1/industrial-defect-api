"""FastAPI entrypoint for industrial defect inspection."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.vision import inspect_bytes

app = FastAPI(
    title="Industrial Defect API",
    description="Portfolio demo: OpenCV defect inspection on synthetic panels",
    version="1.0.0",
)


class InspectResponse(BaseModel):
    result: str
    score: float
    boxes: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)) -> InspectResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        result = inspect_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return InspectResponse(result=result.result, score=result.score, boxes=result.boxes)
