from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import List
import uvicorn
import logging
from contextlib import asynccontextmanager
from utils.inference import batch_inference
from utils.model_utils import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_manager.load_model("models/weights.pt")
        logger.info("Model loaded")
    except Exception:
        logger.exception("Failed to load model")
        raise
    yield


app = FastAPI(title="Stoma Segmentation API", lifespan=lifespan)

app.add_middleware(
    TrustedHostMiddleware,allowed_hosts=[
        "stoma-ml-api-stoma-ml.apps.ocp-test-0.k8s.it.helsinki.fi",
        "stoma-ml-api-stoma-ml.apps.ocp-bm-0.k8s.it.helsinki.fi",
        "localhost",
        "127.0.0.1",
        "backend",
    ],
)
@app.get("/")
async def root():
    return {"status": "healthy", "message": "Stoma Segmentation API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_manager.is_loaded()}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...), conf: float = Query(0.50, ge=0.0, le=1.0), scale: float = Query(10/46, gt=0.0)):
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    images_data = []
    for f in files:
        contents = await f.read()
        if contents:
            images_data.append((f.filename, contents))

    if not images_data:
        raise HTTPException(status_code=400, detail="All uploaded files were empty")

    results = batch_inference(model_manager.get_model(), images_data, conf_threshold=conf, um_per_px=scale)
    cleaned_results = []
    for res in results:
        one_result = {
            "filename": res["filename"],
            "index": res["index"],
            "success": res["success"],
            "density_info": res.get("density_info", {}),
            "stomata": res.get("stomata", []),
        }
        cleaned_results.append(one_result)

    response_data = {
        "total_images": len(results),
        "results": cleaned_results,
    }

    return JSONResponse(content=response_data)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, proxy_headers=True,forwarded_allow_ips="*") 
