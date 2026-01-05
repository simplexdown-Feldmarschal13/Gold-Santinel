from __future__ import annotations

import json
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.config import settings
from app.decision.engine import decide_v1
from app.schemas import AnalysisResponse, ClientContext, VersionInfo
from app.vision.image_io import load_image_bytes
from app.vision.extract import extract_vision

app = FastAPI(title=settings.app_name)

templates = Jinja2Templates(directory="app/web/templates")


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request, "title": "Gold Sentinel"})


@app.get("/app", response_class=HTMLResponse)
def application(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "title": "Application • Gold Sentinel"})


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_v1(
    file: UploadFile = File(...),
    timeframe_hint: str | None = Form(default=None),
    client_context: str | None = Form(default=None),
):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported upload. Provide an image file.")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_mb}MB.")

    bgr = load_image_bytes(data)

    # client_context is accepted per contract; parsed for validation only (no persistence in v1).
    if client_context:
        try:
            ClientContext.model_validate(json.loads(client_context))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid client_context JSON.")

    vision = extract_vision(bgr=bgr)
    decision = decide_v1(vision)

    # Timeframe response per spec: v1 supports hint or unknown (OCR reserved for future hardening).
    if timeframe_hint:
        tf_detected = timeframe_hint
        tf_conf = 1.0
        tf_source = "hint"
    else:
        tf_detected = "unknown"
        tf_conf = 0.0
        tf_source = "unknown"

    return AnalysisResponse(
        analysis_id=str(uuid4()),
        timeframe={"detected": tf_detected, "confidence": tf_conf, "source": tf_source},
        status=decision.status,
        bias=decision.bias,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        features=decision.features,
        safety=decision.safety,
        version=VersionInfo(vision_model=settings.vision_model, decision_model=settings.decision_model),
    )

