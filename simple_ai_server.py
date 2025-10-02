"""
Simplified AI server that works reliably
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import json
import uuid
import time

app = FastAPI(title="Object-Centric AI Demo - Simplified")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo
object_kernels = []

class PredictRequest(BaseModel):
    image_base64: str

class PredictResponse(BaseModel):
    num_kernels: int
    matched_kernel_ids: list
    matched_scores: list
    message: str

@app.get("/")
async def root():
    return {
        "message": "Object-Centric AI Demo Server is running!",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "status": "ready"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        # Decode the image
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        # Get image dimensions for demo
        width, height = img.size
        
        # Create a simple "object kernel" for demo
        kernel_id = str(uuid.uuid4())
        similarity_score = 0.85  # Simulated similarity score
        
        # Add to our simple storage
        object_kernels.append({
            "id": kernel_id,
            "timestamp": time.time(),
            "image_size": f"{width}x{height}",
            "similarity": similarity_score
        })
        
        # Keep only last 10 kernels for demo
        if len(object_kernels) > 10:
            object_kernels.pop(0)
        
        return PredictResponse(
            num_kernels=len(object_kernels),
            matched_kernel_ids=[kernel_id],
            matched_scores=[similarity_score],
            message=f"Successfully processed {width}x{height} image!"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/kernels")
async def get_kernels():
    """Get all stored object kernels"""
    return {"kernels": object_kernels}

if __name__ == "__main__":
    import os
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting simplified AI server on port {port}...")
    print(f"Server will be available at: http://0.0.0.0:{port}")
    print(f"API documentation at: http://0.0.0.0:{port}/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
