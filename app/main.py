from __future__ import annotations

import json
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.config import settings
from app.decision.engine import decide_mtf_v1, decide_v1
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
    file: list[UploadFile] = File(...),
    timeframe_hint: str | None = Form(default=None),
    client_context: str | None = Form(default=None),
):
    if not file:
        raise HTTPException(status_code=400, detail="Missing file upload.")

    for f in file:
        if f.content_type is None or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Unsupported upload. Provide an image file.")

    # client_context is accepted per contract; parsed for validation + optional MTF mode (no persistence in v1).
    ctx_raw: dict | None = None
    if client_context:
        try:
            ctx_raw = json.loads(client_context)
            ClientContext.model_validate(ctx_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid client_context JSON.")

    # ---- Single screenshot fallback (exact v1 behavior) ----
    mtf_mode = bool(isinstance(ctx_raw, dict) and ctx_raw.get("mode") == "MTF")
    if (not mtf_mode) or len(file) != 2:
        data = await file[0].read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_mb}MB.")
        bgr = load_image_bytes(data)
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
    else:
        # ---- MTF mode (two uploads, deterministic ordering) ----
        # Deterministic mapping: file[0] = HTF, file[1] = LTF
        data_htf = await file[0].read()
        data_ltf = await file[1].read()
        if len(data_htf) > settings.max_upload_mb * 1024 * 1024 or len(data_ltf) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_mb}MB.")

        bgr_htf = load_image_bytes(data_htf)
        bgr_ltf = load_image_bytes(data_ltf)

        v_htf = extract_vision(bgr=bgr_htf)
        v_ltf = extract_vision(bgr=bgr_ltf)
        decision = decide_mtf_v1(v_htf, v_ltf)

        # Timeframe in response remains single (v1 contract): use LTF hint if supplied.
        ltf_hint = None
        if isinstance(ctx_raw, dict):
            ltf_hint = (ctx_raw.get("ltf") or {}).get("timeframe_hint")
        tf_detected = ltf_hint or timeframe_hint or "unknown"
        tf_conf = 1.0 if tf_detected != "unknown" else 0.0
        tf_source = "hint" if tf_detected != "unknown" else "unknown"

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

