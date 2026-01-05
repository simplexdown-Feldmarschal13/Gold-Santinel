from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.config import settings
from app.decision.engine import decide, from_structure
from app.schemas import AnalysisResponse, DecisionStatus
from app.vision.image_io import load_image_bytes
from app.vision.plot_region import estimate_plot_bbox
from app.vision.structure import analyze_structure

app = FastAPI(title=settings.app_name)

templates = Jinja2Templates(directory="app/web/templates")


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request, "title": "Gold Sentinel"})


@app.get("/app", response_class=HTMLResponse)
def application(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "title": "Application • Gold Sentinel"})


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), timeframe_hint: str | None = Form(default=None)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported upload. Provide an image file.")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_mb}MB.")

    bgr = load_image_bytes(data)

    plot_bbox = estimate_plot_bbox(gray=_to_gray_for_bbox(bgr))
    struct = analyze_structure(bgr=bgr, plot_bbox=plot_bbox)
    features = from_structure(struct, timeframe=timeframe_hint)

    decision = decide(features)

    # Safety: never claim TRADE when we explicitly lack data.
    if decision.status == DecisionStatus.trade and decision.features.slope is None:
        decision = decision.__class__(
            bias=decision.bias,
            confidence=min(decision.confidence, 40),
            status=DecisionStatus.no_trade,
            reasoning=decision.reasoning + ["Safety override: insufficient structural evidence for trade-eligible output."],
            features=decision.features,
        )

    return AnalysisResponse(
        bias=decision.bias,
        confidence=decision.confidence,
        status=decision.status,
        reasoning=decision.reasoning,
        features=decision.features,
    )


def _to_gray_for_bbox(bgr):
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

