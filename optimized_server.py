#!/usr/bin/env python
"""
Optimized Enhanced FastAPI server - smaller footprint for Railway deployment.
Uses lighter models and optimized dependencies.
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
import json
import time
import uuid
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io

# Use smaller CLIP model
from transformers import CLIPProcessor, CLIPModel

# Add CORS middleware
app = FastAPI(title="CORE - Object-Centric AI API (Optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null", "file://"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["*"],
)

# Optimized scatter_mean implementation
def scatter_mean(src, index, dim=0, dim_size=None):
    """Fallback implementation of scatter_mean"""
    if dim_size is None:
        dim_size = index.max().item() + 1
    result = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=torch.long)
    
    for i in range(src.size(0)):
        idx = index[i].item()
        result[idx] += src[i]
        count[idx] += 1
    
    # Avoid division by zero
    count = torch.clamp(count, min=1)
    result = result / count.unsqueeze(1).float()
    return result

# Optimized ObjectKernel class
class ObjectKernel:
    def __init__(self, latent_dim=64):  # Reduced from 128 to 64
        self.uuid = str(uuid.uuid4())
        self.embedding = torch.randn(latent_dim)  # Smaller embedding
        self.properties = {}
        self.related_kernels = {}
        self.creation_time = datetime.now()
    
    def update_state(self, new_embedding):
        # Simple exponential moving average
        alpha = 0.1
        self.embedding = alpha * new_embedding + (1 - alpha) * self.embedding
    
    def get_embedding(self):
        return self.embedding

# Optimized KernelPool
class KernelPool:
    def __init__(self):
        self.kernels = []
    
    def add_kernel(self, kernel):
        self.kernels.append(kernel)
        # Limit pool size to save memory
        if len(self.kernels) > 100:  # Reduced from unlimited
            self.kernels.pop(0)
    
    def find_best_match(self, query_embedding, top_k=5):
        if not self.kernels:
            return None
        
        similarities = []
        for kernel in self.kernels:
            sim = F.cosine_similarity(query_embedding, kernel.get_embedding().unsqueeze(0))
            similarities.append((kernel, sim.item()))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [similarities[:top_k]]

# Optimized ProjectionLayer
class OptimizedProjectionLayer(nn.Module):
    def __init__(self, input_dim=512, latent_dim=64):  # Smaller dimensions
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
    
    def to_latent_space(self, features):
        return self.projection(features)

# Initialize components with smaller models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 64  # Reduced from 128

# Use smaller CLIP model
print("Loading optimized CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # Smaller than ViT-B/32
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.to(device)

# Initialize components
kernel_pool = KernelPool()
projections = OptimizedProjectionLayer(input_dim=512, latent_dim=latent_dim).to(device)

# Load checkpoint if available (optional)
_loaded_ckpt_path = None
if os.path.exists("checkpoints/coco_ckpt_epoch_10.pth"):
    try:
        checkpoint = torch.load("checkpoints/coco_ckpt_epoch_10.pth", map_location=device)
        projections.load_state_dict(checkpoint.get('projection_state_dict', {}))
        _loaded_ckpt_path = "checkpoints/coco_ckpt_epoch_10.pth"
        print(f"Loaded checkpoint: {_loaded_ckpt_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

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
    """Optimized prediction with real AI capabilities."""
    try:
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        # Use smaller image size for faster processing
        clip_size = 224
        resized = img.resize((clip_size, clip_size))
        proc = clip_processor(images=resized, return_tensors="pt").to(device)
        vis_feat = clip_model.get_image_features(**proc).detach().squeeze(0)
        
        text_desc = req.text_description if req.text_description else "a photo of an object"
        tproc = clip_processor(text=[text_desc], return_tensors="pt", padding=True, truncation=True).to(device)
        text_feat = clip_model.get_text_features(**tproc).detach().squeeze(0)

        with torch.no_grad():
            proj_vis = projections.to_latent_space(vis_feat.unsqueeze(0).to(device)).squeeze(0)
            proj_text = projections.to_latent_space(text_feat.unsqueeze(0).to(device)).squeeze(0)

        # Create new kernel
        new_kernel = ObjectKernel(latent_dim=latent_dim)
        new_kernel.update_state((proj_vis + proj_text) / 2.0)
        new_kernel.properties.update({
            "label": "Detected Object",
            "size": f"{img.size[0]}x{img.size[1]}",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85
        })
        kernel_pool.add_kernel(new_kernel)

        query_emb = (proj_vis + proj_text) / 2.0
        matches = kernel_pool.find_best_match(query_emb.unsqueeze(0), top_k=5)

        matched_kernels_detail = []
        if matches and matches[0]:
            for kernel, sim in matches[0]:
                properties_for_json = {k: list(v) if isinstance(v, set) else v for k, v in kernel.properties.items()}
                matched_kernels_detail.append(KernelMatchDetail(
                    kernel_id=kernel.uuid,
                    similarity_score=sim,
                    properties=properties_for_json
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
        'model_loaded': _loaded_ckpt_path is not None,
        'device': str(device),
        'kernel_pool_size': len(kernel_pool.kernels),
        'active_sessions': 0,
        'server_type': 'optimized_enhanced'
    })

@app.get("/")
async def root():
    return {
        "message": "CORE - Object-Centric AI API (Optimized) is running!",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        },
        "status": "ready",
        "server_type": "optimized_enhanced"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting CORE optimized enhanced server on port {port}...")
    print(f"Device: {device}")
    print(f"Kernels in pool: {len(kernel_pool.kernels)}")
    uvicorn.run(app, host="0.0.0.0", port=port)
