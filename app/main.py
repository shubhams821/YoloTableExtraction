"""
Document Table Extraction API
==============================

Wraps a YOLO (DocLayNet-style) table-detection model + Tabula extraction
pipeline behind a FastAPI service.

Run:
    pip install fastapi uvicorn python-multipart pymupdf opencv-python-headless \
                numpy pandas tabula-py ultralytics
    export YOLO_MODEL_PATH=/path/to/yolov10x_best.pt
    uvicorn app:app --host 0.0.0.0 --port 8000

Note: tabula-py requires a JRE installed on the host.
"""

import os
import shutil
import uuid
from typing import Dict, List, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import tabula
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "data/yolov10x_best.pt")
DPI = int(os.environ.get("PDF_RENDER_DPI", "150"))
CONF_THRESHOLD = float(os.environ.get("TABLE_CONF_THRESHOLD", "0.30"))
TABLE_CLASS_NAME = "table"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "yolov10-doc-v1")

STORAGE_DIR = os.environ.get("STORAGE_DIR", "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}

app = FastAPI(
    title="Document Table Extraction API",
    version="1.0.0",
    description="Detects tables in documents with YOLO and extracts them with Tabula.",
)

# Allow the browser-based frontend (served from a different origin, e.g.
# file:// or a static dev server) to call this API. For production, replace
# allow_origins=["*"] with your actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: document_id -> {"pdf_path": str, "tables": [...]}
# Swap this for Redis/DB in production; it will not survive a restart
# and will not scale across multiple workers.
DOCUMENT_STORE: Dict[str, dict] = {}

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    """Lazily load the YOLO model once per process."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


# ============================================================
# PIPELINE (adapted from the original script)
# ============================================================

def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        pages.append({
            "page_number": page_number,
            "image": image,
            "pdf_width": page.rect.width,
            "pdf_height": page.rect.height,
            "image_width": pix.width,
            "image_height": pix.height,
        })

    doc.close()
    return pages


def image_bbox_to_pdf_bbox(bbox, image_width, image_height, pdf_width, pdf_height):
    x1, y1, x2, y2 = bbox
    scale_x = pdf_width / image_width
    scale_y = pdf_height / image_height
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def detect_tables(model: YOLO, page_data: dict) -> List[dict]:
    image = page_data["image"]
    results = model.predict(source=image, conf=CONF_THRESHOLD, verbose=False)

    detections = []
    result = results[0]
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        if class_name.lower() != TABLE_CLASS_NAME:
            continue

        bbox = box.xyxy[0].cpu().numpy()
        pdf_bbox = image_bbox_to_pdf_bbox(
            bbox,
            page_data["image_width"], page_data["image_height"],
            page_data["pdf_width"], page_data["pdf_height"],
        )

        detections.append({
            "confidence": confidence,
            "image_bbox": bbox.tolist(),
            "pdf_bbox": pdf_bbox,
        })

    return detections


def extract_table_with_tabula(pdf_path: str, page_number: int, pdf_bbox) -> Optional[pd.DataFrame]:
    x1, y1, x2, y2 = pdf_bbox
    area = [y1, x1, y2, x2]  # tabula wants top, left, bottom, right

    try:
        tables = tabula.read_pdf(
            pdf_path,
            pages=page_number,
            area=area,
            guess=False,
            lattice=True,
            multiple_tables=True,
        )
        if tables:
            return tables[0]
    except Exception as e:
        print(f"Tabula failed on page {page_number}: {e}")

    return None


# ============================================================
# SCHEMAS (matches the requested response contract)
# ============================================================

class Element(BaseModel):
    type: str
    bbox: List[float]
    confidence: float


class AnalyzeResponse(BaseModel):
    document_id: str
    elements: List[Element]
    model_version: str


# ============================================================
# ROUTES
# ============================================================

@app.post("/v1/documents/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(..., description="PDF document to analyze"),
    extract_tables: bool = Query(
        False,
        description="If true, also runs Tabula extraction and caches the "
                     "resulting table data for retrieval via the /tables endpoints.",
    ),
):
    """
    Detects table regions in the uploaded document.

    Set extract_tables=true to also extract and cache the actual table
    contents (rows/columns), retrievable afterwards via
    GET /v1/documents/{document_id}/tables.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Only PDF is supported.")

    document_id = str(uuid.uuid4())
    pdf_path = os.path.join(STORAGE_DIR, f"{document_id}.pdf")

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pages = pdf_to_images(pdf_path, dpi=DPI)
    except Exception as e:
        os.remove(pdf_path)
        raise HTTPException(400, f"Failed to read PDF: {e}")

    model = get_model()

    elements: List[dict] = []
    tables_cache: List[dict] = []
    table_counter = 0  # document-wide, NOT reset per page — must match tables_cache position

    for page_data in pages:
        detections = detect_tables(model, page_data)

        for detection in detections:
            bbox = [round(v, 2) for v in detection["pdf_bbox"]]
            confidence = round(detection["confidence"], 4)

            elements.append({
                "type": "table",
                "bbox": bbox,
                "confidence": confidence,
            })

            if extract_tables:
                table_counter += 1

                df = extract_table_with_tabula(
                    pdf_path, page_data["page_number"], detection["pdf_bbox"]
                )
                tables_cache.append({
                    "page": page_data["page_number"],
                    "table_index": table_counter,
                    "bbox": bbox,
                    "confidence": confidence,
                    "dataframe": df,
                })

    DOCUMENT_STORE[document_id] = {
        "pdf_path": pdf_path,
        "tables": tables_cache,
    }

    return {
        "document_id": document_id,
        "elements": elements,
        "model_version": MODEL_VERSION,
    }


@app.get("/v1/documents/{document_id}/tables")
async def list_tables(document_id: str):
    """Lists cached tables for a document (requires extract_tables=true on analyze)."""
    doc = DOCUMENT_STORE.get(document_id)
    if not doc:
        raise HTTPException(404, "document not found")

    return {
        "document_id": document_id,
        "tables": [
            {
                "table_index": t["table_index"],
                "page": t["page"],
                "bbox": t["bbox"],
                "confidence": t["confidence"],
                "rows": int(t["dataframe"].shape[0]) if t["dataframe"] is not None else 0,
                "columns": int(t["dataframe"].shape[1]) if t["dataframe"] is not None else 0,
                "extracted": t["dataframe"] is not None,
            }
            for t in doc["tables"]
        ],
    }


@app.get("/v1/documents/{document_id}/tables/{table_index}")
async def get_table(
    document_id: str,
    table_index: int,
    format: str = Query("json", enum=["json", "csv"]),
):
    """Returns the extracted content of a single table, as JSON rows or CSV."""
    doc = DOCUMENT_STORE.get(document_id)
    if not doc:
        raise HTTPException(404, "document not found")

    if table_index < 1 or table_index > len(doc["tables"]):
        raise HTTPException(404, "table not found")

    entry = doc["tables"][table_index - 1]
    df = entry["dataframe"]

    if df is None:
        raise HTTPException(422, "table extraction failed for this table")

    if format == "csv":
        return Response(content=df.to_csv(index=False), media_type="text/csv")

    return JSONResponse(content=df.fillna("").to_dict(orient="records"))


@app.delete("/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Removes a document and its cached data."""
    doc = DOCUMENT_STORE.pop(document_id, None)
    if not doc:
        raise HTTPException(404, "document not found")

    try:
        os.remove(doc["pdf_path"])
    except OSError:
        pass

    return {"status": "deleted", "document_id": document_id}


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)