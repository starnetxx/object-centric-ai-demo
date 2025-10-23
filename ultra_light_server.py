#!/usr/bin/env python
"""
Ultra-lightweight server - minimal dependencies for Railway free tier.
Uses only basic image processing without heavy ML libraries.
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
app = FastAPI(title="CORE - Object-Centric AI API (Ultra-Light)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null", "file://"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["*"],
)

# Ultra-lightweight mock classes
class MockKernel:
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        self.properties = {
            "label": "Detected Object",
            "size": "224x224",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85
        }
        # Simple hash-based "embedding"
        self.embedding = np.random.rand(32)  # Very small embedding

class MockKernelPool:
    def __init__(self):
        self.kernels = []
        # Create some mock kernels
        for i in range(5):  # Very small pool
            self.kernels.append(MockKernel())
    
    def add_kernel(self, kernel):
        self.kernels.append(kernel)
        # Keep only last 10 kernels
        if len(self.kernels) > 10:
            self.kernels.pop(0)
    
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
    """Ultra-lightweight prediction with mock AI capabilities."""
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
            message=f"Detected object in {img.size[0]}x{img.size[1]} image (Mock AI)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/similarity/search")
async def similarity_search(request: dict):
    """Mock similarity search endpoint."""
    top_matches = []
    candidate_images = request.get('candidate_images_base64', [])
    top_k = request.get('top_k', 5)
    
    for i in range(min(len(candidate_images), top_k)):
        top_matches.append({
            'index': i,
            'similarity': 0.85 - (i * 0.1),
            'image_base64': candidate_images[i] if i < len(candidate_images) else ''
        })
    
    return JSONResponse(content={
        'status': 'success',
        'message': 'Similarity search (mock) - Ultra-light server',
        'top_matches': top_matches,
        'total_matches': len(top_matches),
        'server_type': 'ultra_light'
    })

@app.post("/track/objects")
async def track_objects(request: dict):
    """Mock object tracking endpoint."""
    persistent_objects = [
        {
            'kernel_uuid': str(uuid.uuid4()),
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'appearance_count': 3,
            'properties': {'label': 'Person', 'confidence': 0.92}
        },
        {
            'kernel_uuid': str(uuid.uuid4()),
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'appearance_count': 2,
            'properties': {'label': 'Car', 'confidence': 0.88}
        }
    ]
    
    current_detections = [
        {
            'kernel_id': str(uuid.uuid4()),
            'similarity': 0.92,
            'properties': {'label': 'Person', 'confidence': 0.92}
        },
        {
            'kernel_id': str(uuid.uuid4()),
            'similarity': 0.88,
            'properties': {'label': 'Car', 'confidence': 0.88}
        }
    ]
    
    return JSONResponse(content={
        'status': 'success',
        'message': 'Object tracking (mock) - Ultra-light server',
        'session_summary': {
            'total_unique_objects': 3,
            'persistent_objects': len(persistent_objects),
            'total_observations': 5,
            'session_created': datetime.now().isoformat()
        },
        'persistent_objects': persistent_objects,
        'current_detections': current_detections,
        'server_type': 'ultra_light'
    })

@app.post("/interpret/analyze")
async def analyze_interpretability(request: dict):
    """Mock interpretability analysis endpoint."""
    return JSONResponse(content={
        'status': 'success',
        'message': 'Interpretability analysis (mock) - Ultra-light server',
        'projection_analysis': {
            'effective_rank': 8,
            'projection_quality': 0.85,
            'singular_values': [2.1, 1.8, 1.5, 1.2, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1]
        },
        'kernel_statistics': {
            'total_kernels': len(kernel_pool.kernels),
            'average_confidence': 0.82,
            'confidence_std': 0.15,
            'label_distribution': {
                'Person': 15,
                'Car': 12,
                'Building': 8,
                'Tree': 6,
                'Other': 4
            }
        },
        'kernel_clustering': {
            'pca_variance_explained': [0.35, 0.28, 0.18, 0.12, 0.07],
            'kernel_labels': ['Person', 'Car', 'Building', 'Tree', 'Person', 'Car', 'Building', 'Tree', 'Person', 'Car']
        },
        'server_type': 'ultra_light'
    })

@app.post("/interpret/visualize_kernels")
async def visualize_kernels():
    """Mock kernel visualization endpoint."""
    mock_visualization = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+CiAgPHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzM3NDE1MSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPktlcm5lbCBWaXN1YWxpemF0aW9uPC90ZXh0PgogIDx0ZXh0IHg9IjUwJSIgeT0iNjAlIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2YjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5VbHRyYS1MaWdodCBTZXJ2ZXI8L3RleHQ+Cjwvc3ZnPg=="
    
    return JSONResponse(content={
        'status': 'success',
        'message': 'Kernel visualization (mock) - Ultra-light server',
        'visualization': mock_visualization,
        'kernel_count': len(kernel_pool.kernels),
        'server_type': 'ultra_light'
    })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        'status': 'healthy',
        'model_loaded': False,  # No real model in ultra-light
        'device': 'cpu',
        'kernel_pool_size': len(kernel_pool.kernels),
        'active_sessions': 0,
        'server_type': 'ultra_light'
    })

@app.get("/mobile-test")
async def mobile_test():
    """Mobile-friendly test endpoint."""
    return JSONResponse(content={
        'status': 'mobile_accessible',
        'timestamp': datetime.now().isoformat(),
        'server': 'CORE Object-Centric AI (Ultra-Light)',
        'version': '1.0.0',
        'cors_enabled': True,
        'kernel_pool_size': len(kernel_pool.kernels),
        'device_info': {
            'device': 'cpu',
            'model_loaded': False
        },
        'endpoints_available': [
            '/health', '/predict', '/similarity/search', '/track/objects', 
            '/interpret/analyze', '/interpret/visualize_kernels', '/mobile-test', '/docs'
        ]
    })

@app.get("/")
async def root():
    return {
        "message": "CORE - Object-Centric AI API (Ultra-Light) is running!",
        "endpoints": {
            "predict": "/predict",
            "similarity": "/similarity/search",
            "tracking": "/track/objects",
            "interpretability": "/interpret/analyze",
            "visualize": "/interpret/visualize_kernels",
            "health": "/health",
            "mobile-test": "/mobile-test",
            "docs": "/docs"
        },
        "status": "ready",
        "server_type": "ultra_light"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting CORE ultra-light server on port {port}...")
    print(f"Kernels in pool: {len(kernel_pool.kernels)}")
    uvicorn.run(app, host="0.0.0.0", port=port)
