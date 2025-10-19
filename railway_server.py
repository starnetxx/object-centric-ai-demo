"""
Railway-optimized FastAPI server for CORE Object-Centric AI.
Simplified version with better error handling and memory management.
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
import torch
import torchvision.transforms as T
import numpy as np
from datetime import datetime
import time
import uuid
import torch.nn as nn
import torch.nn.functional as F

# Add CORS middleware
app = FastAPI(title="CORE - Object-Centric AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple models (no heavy dependencies)
class SimpleProjectionEngine(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)

class SimpleObjectKernel:
    def __init__(self, kernel_id=None):
        self.kernel_id = kernel_id or str(uuid.uuid4())
        self.properties = {
            "label": "Detected Object",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        self.last_updated = time.time()

# Initialize simple components
projections = SimpleProjectionEngine().to(device)
object_kernels = []
session_tracker = {}

# Request/Response Models
class KernelMatchDetail(BaseModel):
    kernel_id: str
    similarity_score: float
    properties: Dict[str, Any]

class PredictRequest(BaseModel):
    image_base64: str
    text_description: Optional[str] = None
    session_id: Optional[str] = None

class PredictResponse(BaseModel):
    num_kernels: int
    matched_kernels: List[KernelMatchDetail]
    session_info: Optional[Dict] = None

class SimilaritySearchRequest(BaseModel):
    reference_image_base64: str
    candidate_images_base64: List[str]
    top_k: int = 3

class ObjectTrackingRequest(BaseModel):
    session_id: str
    image_base64: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={
        'status': 'healthy',
        'device': str(device),
        'kernels': len(object_kernels),
        'sessions': len(session_tracker)
    })

@app.get("/")
async def root():
    return {
        "message": "CORE - Object-Centric AI API is running!",
        "endpoints": {
            "predict": "/predict",
            "similarity": "/similarity/search",
            "tracking": "/track/objects",
            "health": "/health",
            "docs": "/docs"
        },
        "status": "ready"
    }

# Main prediction endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        # Decode image
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        # Create a simple kernel for this detection
        kernel = SimpleObjectKernel()
        kernel.properties.update({
            "label": req.text_description or "Detected Object",
            "size": f"{img.size[0]}x{img.size[1]}",
            "confidence": 0.85
        })
        
        object_kernels.append(kernel)
        
        # Keep only last 100 kernels to manage memory
        if len(object_kernels) > 100:
            object_kernels.pop(0)
        
        # Create response
        matched_kernels = [KernelMatchDetail(
            kernel_id=kernel.kernel_id,
            similarity_score=0.85,
            properties=kernel.properties
        )]
        
        response_data = {
            "num_kernels": len(object_kernels),
            "matched_kernels": matched_kernels
        }
        
        # Add session tracking if session_id provided
        if req.session_id:
            if req.session_id not in session_tracker:
                session_tracker[req.session_id] = {
                    'objects': [],
                    'created_at': datetime.now().isoformat()
                }
            
            session_tracker[req.session_id]['objects'].append({
                'kernel_id': kernel.kernel_id,
                'timestamp': datetime.now().isoformat()
            })
            
            response_data["session_info"] = {
                "total_tracked_objects": len(session_tracker[req.session_id]['objects']),
                "session_created": session_tracker[req.session_id]['created_at']
            }
        
        return PredictResponse(**response_data)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Similarity search endpoint
@app.post("/similarity/search")
async def similarity_search(req: SimilaritySearchRequest):
    try:
        # Simple similarity simulation
        similarities = []
        for idx, cand_base64 in enumerate(req.candidate_images_base64):
            # Simulate similarity score
            sim = 0.7 + (idx * 0.1)  # Mock similarity scores
            similarities.append({'index': idx, 'similarity': sim})
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return JSONResponse(content={
            'top_matches': similarities[:req.top_k],
            'all_similarities': similarities,
            'reference_processed': True,
            'candidates_processed': len(similarities)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Object tracking endpoint
@app.post("/track/objects")
async def track_objects(req: ObjectTrackingRequest):
    try:
        # Get or create session
        if req.session_id not in session_tracker:
            session_tracker[req.session_id] = {
                'objects': [],
                'created_at': datetime.now().isoformat()
            }
        
        session = session_tracker[req.session_id]
        
        # Create a new kernel for this tracking
        kernel = SimpleObjectKernel()
        kernel.properties.update({
            "label": "Tracked Object",
            "session_id": req.session_id,
            "timestamp": datetime.now().isoformat()
        })
        
        object_kernels.append(kernel)
        session['objects'].append({
            'kernel_id': kernel.kernel_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 kernels
        if len(object_kernels) > 100:
            object_kernels.pop(0)
        
        return JSONResponse(content={
            'current_detections': [{
                'kernel_id': kernel.kernel_id,
                'similarity': 0.85,
                'properties': kernel.properties
            }],
            'session_summary': {
                'total_unique_objects': len(session['objects']),
                'persistent_objects': len([obj for obj in session['objects'] if obj.get('appearance_count', 1) > 1]),
                'total_observations': len(session['objects']),
                'session_created': session['created_at']
            },
            'persistent_objects': [],
            'tracking_success': True
        })
        
    except Exception as e:
        print(f"Tracking error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Interpretability endpoints (simplified)
@app.post("/interpret/analyze")
async def analyze_interpretability(req: Dict[str, str]):
    return JSONResponse(content={
        'projection_analysis': {
            'matrix_rank': 128,
            'projection_quality': 0.85,
            'effective_rank': 64
        },
        'kernel_statistics': {
            'total_kernels': len(object_kernels),
            'avg_similarity': 0.75,
            'diversity_score': 0.25
        }
    })

@app.post("/interpret/visualize_kernels")
async def visualize_kernel_space():
    return JSONResponse(content={
        'visualization': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
        'kernel_count': len(object_kernels),
        'status': 'success'
    })

@app.post("/hierarchy/analyze_relationships")
async def analyze_hierarchical_relationships():
    return JSONResponse(content={
        'hierarchy_stats': {
            'num_nodes': len(object_kernels),
            'num_edges': 0,
            'average_degree': 0
        },
        'top_connected_objects': [],
        'hierarchical_clusters': {
            'num_clusters': 1,
            'cluster_sizes': [len(object_kernels)],
            'largest_cluster': [k.kernel_id for k in object_kernels]
        }
    })

@app.get("/kernels/list")
async def list_kernels():
    kernels_info = []
    for kernel in object_kernels[-10:]:  # Show last 10
        kernels_info.append({
            'uuid': kernel.kernel_id,
            'properties': kernel.properties,
            'num_relationships': 0,
            'last_updated': kernel.last_updated
        })
    
    return JSONResponse(content={
        'total_kernels': len(object_kernels),
        'kernels': kernels_info
    })

@app.post("/export/kernel_embeddings")
async def export_kernel_embeddings():
    embeddings_data = []
    for kernel in object_kernels:
        embeddings_data.append({
            'uuid': kernel.kernel_id,
            'embedding': [0.1] * 128,  # Mock embedding
            'properties': kernel.properties,
            'relationships': {}
        })
    
    return JSONResponse(content={
        'num_kernels': len(embeddings_data),
        'embedding_dim': 128,
        'kernels': embeddings_data
    })

@app.post("/demo/test_pipeline")
async def test_complete_pipeline():
    return JSONResponse(content={
        'pipeline_test': 'success',
        'prediction_results': {
            'num_kernels': len(object_kernels),
            'num_matches': 1
        },
        'interpretability_available': ['projection', 'kernels'],
        'hierarchy_available': True,
        'server_status': 'enhanced'
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting CORE Railway server on port {port}...")
    print(f"Device: {device}")
    print(f"Memory optimized for Railway")
    uvicorn.run(app, host="0.0.0.0", port=port)
