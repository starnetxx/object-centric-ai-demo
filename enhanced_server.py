#!/usr/bin/env python3
"""
Enhanced FastAPI server with all features, production-ready for Railway deployment.
Simplified version without PyTorch Geometric dependencies.
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

from transformers import CLIPProcessor, CLIPModel

# Add CORS middleware
app = FastAPI(title="CORE - Object-Centric AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple scatter_mean implementation (no torch-scatter needed)
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
    
    count = count.clamp(min=1)
    result = result / count.unsqueeze(1).float()
    return result

class ProjectionEngine(nn.Module):
    """Handles projection between latent space and CLIP space."""
    def __init__(self, latent_dim=256, clip_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.projection_matrix = nn.Parameter(torch.randn(latent_dim, clip_dim) * 0.01)

    def to_clip_space(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.projection_matrix

    def to_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        pinv = torch.linalg.pinv(self.projection_matrix)
        return x @ pinv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent_space(x)

class ObjectKernel(nn.Module):
    """Persistent object identity with GRU-based updates and metadata."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.uuid = str(uuid.uuid4())
        self.latent_dim = latent_dim
        self.state = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.updater = nn.GRUCell(latent_dim, latent_dim)
        self.last_updated = time.time()
        self.properties = {
            "label": None, "affordances": set(), "material": None, "color": None,
            "shape": None, "size": None, "bounding_box": None, "coordinates": None,
        }
        self.related_kernels = {}
        self.update_history = []
        self.activation_history = []

    def forward(self):
        return self.state

    def update_state(self, new_emb: torch.Tensor):
        if new_emb.dim() == 2 and new_emb.shape[0] == 1:
            new_emb = new_emb.squeeze(0)
        if new_emb.numel() == 0:
            return
        with torch.no_grad():
            prev_state = self.state.clone()
            updated = self.updater(new_emb.to(self.state.device), self.state)
            self.state.copy_(updated.detach())
            self.last_updated = time.time()
            self.update_history.append({
                'timestamp': self.last_updated,
                'delta': (updated - prev_state).norm().item()
            })

    def add_relationship(self, relation_type: str, other_uuid: str):
        if relation_type not in self.related_kernels:
            self.related_kernels[relation_type] = []
        if other_uuid not in self.related_kernels[relation_type]:
            self.related_kernels[relation_type].append(other_uuid)

    def get_embedding(self):
        return self.state.detach()

class KernelPool:
    """Manage a pool of ObjectKernel instances with aging/pruning policies."""
    def __init__(self, max_kernels=500, min_age_seconds=60.0, prune_to=400):
        self.kernels = []
        self.max_kernels = max_kernels
        self.min_age_seconds = min_age_seconds
        self.prune_to = prune_to
        self.kernel_activations = {}

    def add_kernel(self, k: ObjectKernel):
        self.kernels.append(k)
        self._prune_if_needed()

    def _prune_if_needed(self):
        if len(self.kernels) <= self.max_kernels:
            return
        self.kernels.sort(key=lambda x: x.last_updated)
        while len(self.kernels) > self.prune_to:
            removed = self.kernels.pop(0)

    def find_best_match(self, query_emb: torch.Tensor, top_k=1, threshold=None):
        if not self.kernels:
            return []
        emb_matrix = torch.stack([k.get_embedding().to(query_emb.device) for k in self.kernels], dim=0)
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        sim = F.cosine_similarity(query_emb.unsqueeze(1), emb_matrix.unsqueeze(0), dim=-1)
        
        if threshold is not None:
            mask = sim >= threshold
            if not mask.any():
                return []
        
        topk_vals, topk_idx = torch.topk(sim, k=min(top_k, emb_matrix.size(0)), dim=1)
        results = []
        for b in range(topk_idx.size(0)):
            batch_results = []
            for i, idx in enumerate(topk_idx[b]):
                if threshold is None or topk_vals[b, i].item() >= threshold:
                    batch_results.append((self.kernels[idx.item()], topk_vals[b, i].item()))
            results.append(batch_results)
        return results

    def get_kernel_statistics(self):
        """Get statistics about kernel pool for interpretability."""
        if not self.kernels:
            return {}
        
        embeddings = torch.stack([k.get_embedding() for k in self.kernels])
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1)
        
        return {
            'total_kernels': len(self.kernels),
            'avg_similarity': sim_matrix.mean().item(),
            'min_similarity': sim_matrix.min().item(),
            'max_similarity': sim_matrix.max().item(),
            'diversity_score': 1.0 - sim_matrix.mean().item(),
        }

    def to_state_dict(self):
        return [{
            "uuid": k.uuid, "state": k.state.detach().cpu(), "properties": k.properties,
            "related_kernels": k.related_kernels, "last_updated": k.last_updated
        } for k in self.kernels]

    def load_from_state_list(self, state_list, device='cpu'):
        self.kernels = []
        for s in state_list:
            k = ObjectKernel(latent_dim=s["state"].numel())
            k.state.data.copy_(s["state"].to(device))
            properties = s.get("properties", {})
            for key, value in properties.items():
                if isinstance(value, list) and key == "affordances":
                     properties[key] = set(value)
            k.properties = properties
            k.related_kernels = s.get("related_kernels", {})
            k.last_updated = s.get("last_updated", time.time())
            self.kernels.append(k)

class SessionTracker:
    """Track objects across multiple inference calls."""
    
    def __init__(self, similarity_threshold=0.85):
        self.sessions = {}
        self.similarity_threshold = similarity_threshold
    
    def get_or_create_session(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'tracked_objects': [],
                'object_history': [],
                'created_at': datetime.now().isoformat()
            }
        return self.sessions[session_id]
    
    def track_object(self, session_id: str, kernel: ObjectKernel, similarity: float, image_id: str):
        """Track an object across frames/images."""
        session = self.get_or_create_session(session_id)
        
        is_new_object = True
        for tracked in session['tracked_objects']:
            if tracked['kernel_uuid'] == kernel.uuid:
                tracked['last_seen'] = datetime.now().isoformat()
                tracked['appearance_count'] += 1
                tracked['similarity_history'].append(similarity)
                is_new_object = False
                break
        
        if is_new_object:
            session['tracked_objects'].append({
                'kernel_uuid': kernel.uuid,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'appearance_count': 1,
                'similarity_history': [similarity],
                'properties': kernel.properties
            })
        
        session['object_history'].append({
            'timestamp': datetime.now().isoformat(),
            'image_id': image_id,
            'kernel_uuid': kernel.uuid,
            'similarity': similarity
        })

# Initialize components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
clip_dim = 512
projections = ProjectionEngine(latent_dim=latent_dim, clip_dim=clip_dim).to(device)
kernel_pool = KernelPool(max_kernels=500)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
session_tracker = SessionTracker()

# Load checkpoint if available
def load_checkpoint_for_inference(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f"[server] checkpoint not found: {checkpoint_path}")
        return None
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        projections.load_state_dict(state["projections"])
        kp_state = state.get("kernel_pool", [])
        kernel_pool.load_from_state_list(kp_state, device=device)
        print(f"[server] loaded checkpoint {checkpoint_path} with {len(kernel_pool.kernels)} kernels")
        return checkpoint_path
    except Exception as e:
        print(f"[server] failed to load checkpoint {checkpoint_path}: {e}")
        return None

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "coco_ckpt_epoch_10.pth")
_loaded_ckpt_path = load_checkpoint_for_inference(CHECKPOINT_PATH, device=device)

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

# Main prediction endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Enhanced prediction with optional session tracking."""
    try:
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        clip_size = 224  # CLIP ViT-B/32 uses 224x224 images
        resized = img.resize((clip_size, clip_size))
        proc = clip_processor(images=resized, return_tensors="pt").to(device)
        vis_feat = clip_model.get_image_features(**proc).detach().squeeze(0)
        
        text_desc = req.text_description if req.text_description else "a photo of an object"
        tproc = clip_processor(text=[text_desc], return_tensors="pt", padding=True, truncation=True).to(device)
        text_feat = clip_model.get_text_features(**tproc).detach().squeeze(0)

        with torch.no_grad():
            proj_vis = projections.to_latent_space(vis_feat.unsqueeze(0).to(device)).squeeze(0)
            proj_text = projections.to_latent_space(text_feat.unsqueeze(0).to(device)).squeeze(0)

        # Always create a new kernel for each detection
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
                
                if req.session_id:
                    session_tracker.track_object(req.session_id, kernel, sim, 
                                                image_id=f"img_{datetime.now().timestamp()}")
        
        # Ensure we always return at least one result
        if not matched_kernels_detail:
            properties_for_json = {k: list(v) if isinstance(v, set) else v for k, v in new_kernel.properties.items()}
            matched_kernels_detail.append(KernelMatchDetail(
                kernel_id=new_kernel.uuid,
                similarity_score=1.0,
                properties=properties_for_json
            ))

        response_data = {
            "num_kernels": len(kernel_pool.kernels),
            "matched_kernels": matched_kernels_detail
        }
        
        if req.session_id:
            session = session_tracker.get_or_create_session(req.session_id)
            response_data["session_info"] = {
                "total_tracked_objects": len(session['tracked_objects']),
                "history_length": len(session['object_history'])
            }

        return PredictResponse(**response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Similarity search endpoint
@app.post("/similarity/search")
async def similarity_search(req: SimilaritySearchRequest):
    """Find most similar images from a set of candidates."""
    try:
        # Process reference image
        ref_data = base64.b64decode(req.reference_image_base64)
        ref_img = Image.open(BytesIO(ref_data)).convert("RGB")
        clip_size = 224  # CLIP ViT-B/32 uses 224x224 images
        ref_resized = ref_img.resize((clip_size, clip_size))
        ref_proc = clip_processor(images=ref_resized, return_tensors="pt").to(device)
        ref_feat = clip_model.get_image_features(**ref_proc).detach().squeeze(0)
        
        with torch.no_grad():
            ref_latent = projections.to_latent_space(ref_feat.unsqueeze(0).to(device)).squeeze(0)
        
        # Process candidate images
        similarities = []
        for idx, cand_base64 in enumerate(req.candidate_images_base64):
            cand_data = base64.b64decode(cand_base64)
            cand_img = Image.open(BytesIO(cand_data)).convert("RGB")
            cand_resized = cand_img.resize((clip_size, clip_size))
            cand_proc = clip_processor(images=cand_resized, return_tensors="pt").to(device)
            cand_feat = clip_model.get_image_features(**cand_proc).detach().squeeze(0)
            
            with torch.no_grad():
                cand_latent = projections.to_latent_space(cand_feat.unsqueeze(0).to(device)).squeeze(0)
            
            sim = F.cosine_similarity(ref_latent.unsqueeze(0), cand_latent.unsqueeze(0)).item()
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
    """Track objects across multiple images in a session."""
    try:
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        clip_size = 224  # CLIP ViT-B/32 uses 224x224 images
        resized = img.resize((clip_size, clip_size))
        proc = clip_processor(images=resized, return_tensors="pt").to(device)
        vis_feat = clip_model.get_image_features(**proc).detach().squeeze(0)
        
        with torch.no_grad():
            proj_vis = projections.to_latent_space(vis_feat.unsqueeze(0).to(device)).squeeze(0)
        
        matches = kernel_pool.find_best_match(proj_vis.unsqueeze(0), top_k=10)
        
        tracked_objects = []
        if matches and matches[0]:
            for kernel, sim in matches[0]:
                session_tracker.track_object(
                    req.session_id, 
                    kernel, 
                    sim,
                    image_id=f"img_{datetime.now().timestamp()}"
                )
                tracked_objects.append({
                    'kernel_id': kernel.uuid,
                    'similarity': sim,
                    'properties': {k: list(v) if isinstance(v, set) else v 
                                 for k, v in kernel.properties.items()}
                })
        
        session = session_tracker.get_or_create_session(req.session_id)
        persistent_objects = [
            obj for obj in session['tracked_objects'] 
            if obj['appearance_count'] > 1
        ]
        
        return JSONResponse(content={
            'current_detections': tracked_objects,
            'session_summary': {
                'total_unique_objects': len(session['tracked_objects']),
                'persistent_objects': len(persistent_objects),
                'total_observations': len(session['object_history']),
                'session_created': session['created_at']
            },
            'persistent_objects': persistent_objects[:5],
            'tracking_success': True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Interpretability endpoints
@app.post("/interpret/analyze")
async def analyze_interpretability(req: Dict[str, str]):
    """Analyze model interpretability."""
    analysis_type = req.get("analysis_type", "all")
    results = {}
    
    if analysis_type in ["projection", "all"]:
        with torch.no_grad():
            proj_matrix = projections.projection_matrix.cpu().numpy()
            U, S, Vt = np.linalg.svd(proj_matrix, full_matrices=False)
            explained_variance = S**2 / np.sum(S**2)
            cumulative_variance = np.cumsum(explained_variance)
            
            results["projection_analysis"] = {
                'singular_values': [float(x) for x in S.tolist()[:10]],
                'explained_variance_ratio': [float(x) for x in explained_variance.tolist()[:10]],
                'cumulative_variance': [float(x) for x in cumulative_variance.tolist()[:10]],
                'effective_rank': int(np.sum(S > 0.01)),
                'projection_quality': float(cumulative_variance[min(10, len(cumulative_variance)-1)]),
                'matrix_rank': int(np.linalg.matrix_rank(proj_matrix)),
                'condition_number': float(np.linalg.cond(proj_matrix))
            }
    
    if analysis_type in ["kernels", "all"]:
        results["kernel_statistics"] = kernel_pool.get_kernel_statistics()
    
    return JSONResponse(content=results)

@app.post("/interpret/visualize_kernels")
async def visualize_kernel_space():
    """Generate visualization of kernel space."""
    try:
        if len(kernel_pool.kernels) < 2:
            # Create a simple placeholder visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.text(0.5, 0.5, f'Not enough kernels for visualization\n(Need at least 2, have {len(kernel_pool.kernels)})', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Kernel Space Visualization')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return JSONResponse(content={
                'visualization': f'data:image/png;base64,{img_base64}',
                'kernel_count': len(kernel_pool.kernels),
                'status': 'insufficient_data'
            })
        
        embeddings = torch.stack([k.get_embedding() for k in kernel_pool.kernels])
        embeddings_np = embeddings.cpu().numpy()
        
        if len(kernel_pool.kernels) >= 3:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings_np)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
            ax.set_title('Kernel Space (PCA)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return JSONResponse(content={
                'visualization': f'data:image/png;base64,{img_base64}',
                'kernel_count': len(kernel_pool.kernels),
                'status': 'success'
            })
        else:
            # Create a simple 2D plot for 2 kernels
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.scatter([0, 1], [0, 1], alpha=0.6, s=100)
            ax.set_title('Kernel Space (2 Kernels)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return JSONResponse(content={
                'visualization': f'data:image/png;base64,{img_base64}',
                'kernel_count': len(kernel_pool.kernels),
                'status': 'limited_data'
            })
    
    except Exception as e:
        return JSONResponse(content={
            'error': str(e),
            'kernel_count': len(kernel_pool.kernels),
            'status': 'error'
        })

# Hierarchy analysis endpoints
@app.post("/hierarchy/analyze_relationships")
async def analyze_hierarchical_relationships():
    """Analyze hierarchical relationships between objects."""
    try:
        relationship_counts = {}
        for kernel in kernel_pool.kernels:
            rel_count = sum(len(related) for related in kernel.related_kernels.values())
            relationship_counts[kernel.uuid] = {
                'count': rel_count,
                'types': list(kernel.related_kernels.keys()),
                'label': kernel.properties.get('label', 'unknown')
            }
        
        top_connected = sorted(relationship_counts.items(), 
                              key=lambda x: x[1]['count'], 
                              reverse=True)[:10]
        
        return JSONResponse(content={
            'hierarchy_stats': {
                'num_nodes': len(kernel_pool.kernels),
                'num_edges': sum(rel_count for rel_count in relationship_counts.values()),
                'average_degree': sum(rel_count for rel_count in relationship_counts.values()) / len(kernel_pool.kernels) if kernel_pool.kernels else 0
            },
            'top_connected_objects': top_connected,
            'hierarchical_clusters': {
                'num_clusters': 1,
                'cluster_sizes': [len(kernel_pool.kernels)],
                'largest_cluster': [k.uuid for k in kernel_pool.kernels]
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kernels/list")
async def list_kernels():
    """List all kernels with their properties."""
    kernels_info = []
    for kernel in kernel_pool.kernels:
        properties_clean = {k: list(v) if isinstance(v, set) else v 
                          for k, v in kernel.properties.items()}
        kernels_info.append({
            'uuid': kernel.uuid,
            'properties': properties_clean,
            'num_relationships': sum(len(rels) for rels in kernel.related_kernels.values()),
            'last_updated': kernel.last_updated
        })
    
    return JSONResponse(content={
        'total_kernels': len(kernel_pool.kernels),
        'kernels': kernels_info
    })

@app.post("/export/kernel_embeddings")
async def export_kernel_embeddings():
    """Export kernel embeddings for external analysis."""
    if not kernel_pool.kernels:
        raise HTTPException(status_code=400, detail="No kernels available")
    
    embeddings_data = []
    for kernel in kernel_pool.kernels:
        embeddings_data.append({
            'uuid': kernel.uuid,
            'embedding': kernel.get_embedding().cpu().numpy().tolist(),
            'properties': {k: list(v) if isinstance(v, set) else v 
                         for k, v in kernel.properties.items()},
            'relationships': kernel.related_kernels
        })
    
    return JSONResponse(content={
        'num_kernels': len(embeddings_data),
        'embedding_dim': latent_dim,
        'kernels': embeddings_data
    })

@app.post("/demo/test_pipeline")
async def test_complete_pipeline():
    """Demo endpoint to test the complete pipeline."""
    try:
        # Create a simple test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        buf = io.BytesIO()
        test_img.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        
        # Test prediction
        pred_req = PredictRequest(
            image_base64=img_base64,
            text_description="a test object",
            session_id="demo_session"
        )
        pred_response = await predict(pred_req)
        
        # Test interpretability
        interpret_req = {"analysis_type": "all"}
        interpret_response = await analyze_interpretability(interpret_req)
        
        return JSONResponse(content={
            'pipeline_test': 'success',
            'prediction_results': {
                'num_kernels': pred_response.num_kernels,
                'num_matches': len(pred_response.matched_kernels)
            },
            'interpretability_results': interpret_response.body.decode() if hasattr(interpret_response, 'body') else 'available',
            'interpretability_available': ['projection', 'kernels'],
            'hierarchy_available': True,
            'test_image_base64': img_base64,
            'server_status': 'enhanced'
        })
        
    except Exception as e:
        import traceback
        return JSONResponse(content={
            'pipeline_test': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'server_status': 'error'
        })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        'status': 'healthy',
        'model_loaded': _loaded_ckpt_path is not None,
        'device': str(device),
        'kernel_pool_size': len(kernel_pool.kernels),
        'active_sessions': len(session_tracker.sessions)
    })

@app.get("/")
async def root():
    return {
        "message": "CORE - Object-Centric AI API is running!",
        "endpoints": {
            "predict": "/predict",
            "similarity": "/similarity/search",
            "tracking": "/track/objects",
            "interpretability": "/interpret/analyze",
            "hierarchy": "/hierarchy/analyze_relationships",
            "kernels": "/kernels/list",
            "export": "/export/kernel_embeddings",
            "demo": "/demo/test_pipeline",
            "health": "/health",
            "docs": "/docs"
        },
        "status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting CORE enhanced server on port {port}...")
    print(f"Device: {device}")
    print(f"Checkpoint loaded: {_loaded_ckpt_path}")
    print(f"Kernels in pool: {len(kernel_pool.kernels)}")
    uvicorn.run(app, host="0.0.0.0", port=port)
