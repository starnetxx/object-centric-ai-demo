#!/usr/bin/env python
"""
Lightweight FastAPI server for Railway deployment - minimal dependencies
"""
import os
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
from datetime import datetime
import json
import time
import uuid

# Add CORS middleware
app = FastAPI(title="CORE - Object-Centric AI API (Lightweight)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null", "file://"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["*"],
)

# Lightweight mock classes for demo purposes
class MockKernel:
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        self.properties = {
            "label": "Mock Object",
            "size": "224x224",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85
        }
        self.embedding = np.random.rand(128)

class MockKernelPool:
    def __init__(self):
        self.kernels = []
        # Create some mock kernels
        for i in range(10):
            self.kernels.append(MockKernel())
    
    def add_kernel(self, kernel):
        self.kernels.append(kernel)
    
    def find_best_match(self, query_emb, top_k=5):
        if not self.kernels:
            return None
        # Return mock matches
        matches = []
        for kernel in self.kernels[:top_k]:
            matches.append((kernel, 0.85))
        return [matches]

# Initialize mock components
kernel_pool = MockKernelPool()

# Request/Response models
class PredictRequest(BaseModel):
    image_base64: str
    text_description: Optional[str] = None
    session_id: Optional[str] = None

class KernelMatchDetail(BaseModel):
    kernel_id: str
    similarity_score: float
    properties: Dict[str, Any]

class PredictResponse(BaseModel):
    num_kernels: int
    matched_kernels: List[KernelMatchDetail]
    message: str

# Main prediction endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Lightweight prediction endpoint."""
    try:
        # Decode image
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        # Create a new mock kernel
        new_kernel = MockKernel()
        new_kernel.properties.update({
            "label": "Detected Object",
            "size": f"{img.size[0]}x{img.size[1]}",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85
        })
        kernel_pool.add_kernel(new_kernel)

        # Return mock matches
        matched_kernels_detail = []
        for kernel in kernel_pool.kernels[-5:]:  # Last 5 kernels
            matched_kernels_detail.append(KernelMatchDetail(
                kernel_id=kernel.uuid,
                similarity_score=0.85,
                properties=kernel.properties
            ))

        return PredictResponse(
            num_kernels=len(kernel_pool.kernels),
            matched_kernels=matched_kernels_detail,
            message=f"Detected object in {img.size[0]}x{img.size[1]} image"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        'status': 'healthy',
        'model_loaded': True,
        'device': 'cpu',
        'kernel_pool_size': len(kernel_pool.kernels),
        'active_sessions': 0,
        'server_type': 'lightweight'
    })

@app.post("/similarity/search")
async def similarity_search(request: dict):
    """Mock similarity search endpoint."""
    return JSONResponse(content={
        'status': 'success',
        'message': 'Similarity search (mock) - Lightweight server',
        'results': [
            {
                'kernel_id': str(uuid.uuid4()),
                'similarity_score': 0.85,
                'properties': {
                    'label': 'Similar Object',
                    'confidence': 0.85,
                    'timestamp': datetime.now().isoformat()
                }
            }
        ],
        'total_matches': 1,
        'server_type': 'lightweight'
    })

@app.post("/track/objects")
async def track_objects(request: dict):
    """Mock object tracking endpoint."""
    return JSONResponse(content={
        'status': 'success',
        'message': 'Object tracking (mock) - Lightweight server',
        'session_summary': {
            'total_unique_objects': 3,
            'persistent_objects': 2,
            'total_observations': 5,
            'session_created': datetime.now().isoformat()
        },
        'tracked_objects': [
            {
                'kernel_uuid': str(uuid.uuid4()),
                'first_seen': datetime.now().isoformat(),
                'appearance_count': 2,
                'properties': {'label': 'Tracked Object'}
            }
        ],
        'server_type': 'lightweight'
    })

@app.post("/interpret/analyze")
async def analyze_interpretability(request: dict):
    """Mock interpretability analysis endpoint."""
    return JSONResponse(content={
        'status': 'success',
        'message': 'Interpretability analysis (mock) - Lightweight server',
        'analysis_results': {
            'projection_available': True,
            'kernels_available': True,
            'hierarchy_available': True
        },
        'interpretability_available': ['projection', 'kernels'],
        'hierarchy_available': True,
        'server_type': 'lightweight'
    })

@app.get("/mobile-test")
async def mobile_test():
    """Mobile-friendly test endpoint."""
    return JSONResponse(content={
        'status': 'mobile_accessible',
        'timestamp': datetime.now().isoformat(),
        'server': 'CORE Object-Centric AI (Lightweight)',
        'version': '1.0.0',
        'cors_enabled': True,
        'kernel_pool_size': len(kernel_pool.kernels),
        'device_info': {
            'device': 'cpu',
            'model_loaded': True
        },
        'endpoints_available': [
            '/health', '/predict', '/similarity/search', '/track/objects', 
            '/interpret/analyze', '/mobile-test', '/docs'
        ]
    })

@app.get("/")
async def root():
    return {
        "message": "CORE - Object-Centric AI API (Lightweight) is running!",
        "endpoints": {
            "predict": "/predict",
            "similarity": "/similarity/search",
            "tracking": "/track/objects",
            "interpretability": "/interpret/analyze",
            "health": "/health",
            "mobile-test": "/mobile-test",
            "docs": "/docs"
        },
        "status": "ready",
        "server_type": "lightweight"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting CORE lightweight server on port {port}...")
    print(f"Kernels in pool: {len(kernel_pool.kernels)}")
    uvicorn.run(app, host="0.0.0.0", port=port)
