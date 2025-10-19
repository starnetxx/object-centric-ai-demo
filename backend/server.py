"""
Enhanced FastAPI server with interpretability, object tracking, and analysis endpoints.
Works with existing trained checkpoint.
"""

import os
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from datetime import datetime
import json

from transformers import CLIPProcessor, CLIPModel
import time
import uuid
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import torch_geometric.nn as geo_nn

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io

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
        # Add tracking history for interpretability
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
            # Store previous state for interpretability
            prev_state = self.state.clone()
            updated = self.updater(new_emb.to(self.state.device), self.state)
            self.state.copy_(updated.detach())
            self.last_updated = time.time()
            # Track update history
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
        # Add tracking for interpretability
        self.kernel_activations = {}  # Track which kernels activate for which objects

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
        
        # Apply threshold if specified
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
        
        # Compute pairwise similarities
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1)
        
        # Get kernel clustering
        return {
            'total_kernels': len(self.kernels),
            'avg_similarity': sim_matrix.mean().item(),
            'min_similarity': sim_matrix.min().item(),
            'max_similarity': sim_matrix.max().item(),
            'diversity_score': 1.0 - sim_matrix.mean().item(),  # Higher is more diverse
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

class EdgeAggregator(nn.Module):
    def __init__(self, edge_dim, node_count_estimate=128):
        super().__init__()
        self.edge_dim = edge_dim

    def forward(self, edge_index, edge_attr, num_nodes):
        if edge_attr is None or edge_attr.numel() == 0:
            return torch.zeros(num_nodes, 0, device=edge_index.device)
        target_nodes = edge_index[1]
        node_edge_feats = scatter_mean(edge_attr, target_nodes, dim=0, dim_size=num_nodes)
        return node_edge_feats

class GNNModel(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=256, out_dim=256, edge_dim=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
            gat_in_dim = hidden_dim * 2
        else:
            self.edge_proj = None
            gat_in_dim = hidden_dim

        self.gat1 = geo_nn.GATConv(gat_in_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat2 = geo_nn.GATConv(hidden_dim, out_dim // num_heads, heads=num_heads, concat=True)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, node_feats, edge_index, edge_attr=None):
        N = node_feats.size(0)
        x = self.node_proj(node_feats)

        if self.edge_dim > 0 and edge_attr is not None and edge_attr.numel() > 0:
            node_edge_feats = EdgeAggregator(self.edge_dim)(edge_index, edge_attr, num_nodes=N)
            node_edge_feats = self.edge_proj(node_edge_feats)
            x = torch.cat([x, node_edge_feats], dim=-1)
        else:
            x = torch.cat([x, torch.zeros_like(x)], dim=-1) if self.edge_dim > 0 else x

        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.out_proj(x)
        return x

class HierarchyEngine(nn.Module):
    def __init__(self, latent_dim=256, edge_type_list=None, gnn_hidden=256, gnn_out=256, edge_emb_dim=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.edge_type_list = edge_type_list or ["support:on", "support:under", "articulation:hinge", "articulation:slider", "relation:next_to"]
        self.edge_type_to_idx = {t: i for i, t in enumerate(self.edge_type_list)}
        self.edge_emb_dim = edge_emb_dim

        self.edge_type_embedding = nn.Embedding(len(self.edge_type_list), edge_emb_dim)
        self.gnn = GNNModel(node_in_dim=latent_dim, hidden_dim=gnn_hidden, out_dim=gnn_out, edge_dim=edge_emb_dim, num_heads=num_heads, dropout=dropout)

    def _encode_edges(self, kernels_list):
        uuid_to_idx = {k.uuid: i for i, k in enumerate(kernels_list)}
        srcs = []
        dsts = []
        edge_attrs = []
        for i, k in enumerate(kernels_list):
            for rel_type, related in k.related_kernels.items():
                for target_uuid in related:
                    if target_uuid in uuid_to_idx:
                        j = uuid_to_idx[target_uuid]
                        srcs.append(i)
                        dsts.append(j)
                        rel_key = rel_type if rel_type in self.edge_type_to_idx else "relation:next_to"
                        idx = self.edge_type_to_idx.get(rel_key, 0)
                        edge_attrs.append(idx)

        if not srcs:
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, self.edge_emb_dim)

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_idx_tensor = torch.tensor(edge_attrs, dtype=torch.long)
        edge_attr = self.edge_type_embedding(edge_idx_tensor)
        return edge_index, edge_attr

    def compute_hierarchical_embeddings(self, kernel_pool, device=None):
        kernels_list = kernel_pool.kernels
        K = len(kernels_list)
        if K == 0:
            return torch.empty(0, self.gnn.out_proj.out_features if hasattr(self.gnn, 'out_proj') else self.gnn.hidden_dim)

        node_feats = torch.stack([k.get_embedding() for k in kernels_list], dim=0)
        node_feats = node_feats.to(device) if device is not None else node_feats

        edge_index, edge_attr = self._encode_edges(kernels_list)
        if edge_index.numel() == 0:
            edge_index = edge_index.to(node_feats.device)
            edge_attr = torch.empty(0, self.edge_emb_dim, device=node_feats.device)
        else:
            edge_index = edge_index.to(node_feats.device)
            edge_attr = edge_attr.to(node_feats.device)

        hierarchical_embeddings = self.gnn(node_feats, edge_index, edge_attr)
        return hierarchical_embeddings


class InterpretabilityEngine:
    """Extract interpretability insights from the existing trained model."""
    
    def __init__(self, projections, hierarchy, kernel_pool, clip_model, device):
        self.projections = projections
        self.hierarchy = hierarchy
        self.kernel_pool = kernel_pool
        self.clip_model = clip_model
        self.device = device
        
    def analyze_projection_space(self):
        """Analyze the learned projection matrix for interpretability."""
        with torch.no_grad():
            proj_matrix = self.projections.projection_matrix.cpu().numpy()
            
            # Compute SVD to understand principal components
            U, S, Vt = np.linalg.svd(proj_matrix, full_matrices=False)
            
            # Analyze information preservation
            explained_variance = S**2 / np.sum(S**2)
            cumulative_variance = np.cumsum(explained_variance)
            
            return {
                'singular_values': S.tolist()[:10],  # Top 10 singular values
                'explained_variance_ratio': explained_variance.tolist()[:10],
                'cumulative_variance': cumulative_variance.tolist()[:10],
                'effective_rank': int(np.sum(S > 0.01)),  # Effective rank
                'projection_quality': float(cumulative_variance[min(10, len(cumulative_variance)-1)]),
            }
    
    def analyze_kernel_clustering(self):
        """Analyze how kernels cluster in the learned space."""
        if len(self.kernel_pool.kernels) < 2:
            return {'error': 'Not enough kernels for clustering analysis'}
        
        with torch.no_grad():
            embeddings = torch.stack([k.get_embedding() for k in self.kernel_pool.kernels])
            embeddings_np = embeddings.cpu().numpy()
            
            # Perform dimensionality reduction for visualization
            if len(self.kernel_pool.kernels) > 3:
                pca = PCA(n_components=min(3, len(self.kernel_pool.kernels)))
                pca_result = pca.fit_transform(embeddings_np)
                
                # TSNE for 2D visualization (if enough samples)
                if len(self.kernel_pool.kernels) >= 10:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.kernel_pool.kernels)-1))
                    tsne_result = tsne.fit_transform(embeddings_np)
                else:
                    tsne_result = pca_result[:, :2]
                
                return {
                    'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
                    'pca_coords': pca_result.tolist(),
                    'tsne_coords': tsne_result.tolist(),
                    'kernel_labels': [k.properties.get('label', 'unknown') for k in self.kernel_pool.kernels],
                    'kernel_uuids': [k.uuid for k in self.kernel_pool.kernels],
                }
            else:
                return {
                    'raw_embeddings': embeddings_np.tolist(),
                    'kernel_labels': [k.properties.get('label', 'unknown') for k in self.kernel_pool.kernels],
                }
    
    def analyze_hierarchy_graph(self):
        """Analyze the hierarchical relationships in the kernel pool."""
        if not self.kernel_pool.kernels:
            return {'error': 'No kernels available'}
        
        # Build adjacency information
        relationships = {}
        for kernel in self.kernel_pool.kernels:
            if kernel.related_kernels:
                relationships[kernel.uuid] = kernel.related_kernels
        
        # Analyze graph structure
        num_edges = sum(len(rels) for rels in relationships.values())
        num_nodes = len(self.kernel_pool.kernels)
        
        # Find connected components
        visited = set()
        components = []
        
        def dfs(node_uuid, component):
            if node_uuid in visited:
                return
            visited.add(node_uuid)
            component.append(node_uuid)
            kernel = next((k for k in self.kernel_pool.kernels if k.uuid == node_uuid), None)
            if kernel and kernel.related_kernels:
                for rel_type, related_uuids in kernel.related_kernels.items():
                    for uuid in related_uuids:
                        dfs(uuid, component)
        
        for kernel in self.kernel_pool.kernels:
            if kernel.uuid not in visited:
                component = []
                dfs(kernel.uuid, component)
                components.append(component)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_components': len(components),
            'largest_component_size': max(len(c) for c in components) if components else 0,
            'average_degree': (2 * num_edges / num_nodes) if num_nodes > 0 else 0,
            'relationships_summary': {
                rel_type: sum(1 for k in self.kernel_pool.kernels 
                             if rel_type in k.related_kernels)
                for rel_type in set().union(*[set(k.related_kernels.keys()) 
                                             for k in self.kernel_pool.kernels])
            } if relationships else {}
        }
    
    def compute_attention_maps(self, image_tensor, text_tensor):
        """Compute attention-like scores between image regions and text."""
        # This simulates attention maps using similarity scores
        with torch.no_grad():
            # Project to latent space
            img_latent = self.projections.to_latent_space(image_tensor)
            text_latent = self.projections.to_latent_space(text_tensor)
            
            # Compute cross-attention scores
            attention_scores = F.softmax(
                torch.matmul(img_latent, text_latent.T) / np.sqrt(img_latent.shape[-1]),
                dim=-1
            )
            
            return attention_scores.cpu().numpy()

# ============= Session tracking for object persistence =============

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
        
        # Check if this is a previously seen object
        is_new_object = True
        for tracked in session['tracked_objects']:
            if tracked['kernel_uuid'] == kernel.uuid:
                # Update existing object
                tracked['last_seen'] = datetime.now().isoformat()
                tracked['appearance_count'] += 1
                tracked['similarity_history'].append(similarity)
                is_new_object = False
                break
        
        if is_new_object:
            # Add new object to tracking
            session['tracked_objects'].append({
                'kernel_uuid': kernel.uuid,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'appearance_count': 1,
                'similarity_history': [similarity],
                'properties': kernel.properties
            })
        
        # Add to history
        session['object_history'].append({
            'timestamp': datetime.now().isoformat(),
            'image_id': image_id,
            'kernel_uuid': kernel.uuid,
            'similarity': similarity
        })

# ============= Initialize components =============

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

app = FastAPI(title="Object-Centric AI API with Interpretability")

# Globals
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
clip_dim = 512
projections = ProjectionEngine(latent_dim=latent_dim, clip_dim=clip_dim).to(device)
hierarchy = HierarchyEngine(latent_dim=latent_dim, gnn_out=256).to(device)
kernel_pool = KernelPool(max_kernels=500)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# New components
interpretability_engine = InterpretabilityEngine(projections, hierarchy, kernel_pool, clip_model, device)
session_tracker = SessionTracker()

# Load checkpoint
def load_checkpoint_for_inference(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f"[server] checkpoint not found: {checkpoint_path}")
        return None
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        projections.load_state_dict(state["projections"])
        if state.get("hierarchy_gnn") is not None:
            hierarchy.gnn.load_state_dict(state["hierarchy_gnn"])
        kp_state = state.get("kernel_pool", [])
        kernel_pool.load_from_state_list(kp_state, device=device)
        print(f"[server] loaded checkpoint {checkpoint_path} with {len(kernel_pool.kernels)} kernels")
        return checkpoint_path
    except Exception as e:
        print(f"[server] failed to load checkpoint {checkpoint_path}: {e}")
        return None

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "coco_ckpt_epoch_10.pth")
_loaded_ckpt_path = load_checkpoint_for_inference(CHECKPOINT_PATH, device=device)

# ============= Request/Response Models =============

class KernelMatchDetail(BaseModel):
    kernel_id: str
    similarity_score: float
    properties: Dict[str, Any]

class PredictRequest(BaseModel):
    image_base64: str
    text_description: Optional[str] = None
    session_id: Optional[str] = None  # For object tracking

class PredictResponse(BaseModel):
    num_kernels: int
    matched_kernels: List[KernelMatchDetail]
    session_info: Optional[Dict] = None

class InterpretabilityRequest(BaseModel):
    analysis_type: str  # "projection", "clustering", "hierarchy", "all"

class SimilaritySearchRequest(BaseModel):
    reference_image_base64: str
    candidate_images_base64: List[str]
    top_k: int = 3

class ObjectTrackingRequest(BaseModel):
    session_id: str
    image_base64: str
    
# ============= Original Endpoint (Enhanced) =============

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Enhanced prediction with optional session tracking."""
    if _loaded_ckpt_path is None:
        raise HTTPException(status_code=503, detail="Model checkpoint not loaded.")

    try:
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        clip_size = clip_processor.model.config.vision_config.image_size
        resized = img.resize((clip_size, clip_size))
        proc = clip_processor(images=resized, return_tensors="pt").to(device)
        vis_feat = clip_model.get_image_features(**proc).detach().squeeze(0)
        
        text_desc = req.text_description if req.text_description else "a photo of an object"
        tproc = clip_processor(text=[text_desc], return_tensors="pt", padding=True, truncation=True).to(device)
        text_feat = clip_model.get_text_features(**tproc).detach().squeeze(0)

        with torch.no_grad():
            proj_vis = projections.to_latent_space(vis_feat.unsqueeze(0).to(device)).squeeze(0)
            proj_text = projections.to_latent_space(text_feat.unsqueeze(0).to(device)).squeeze(0)

        if len(kernel_pool.kernels) == 0:
            raise HTTPException(status_code=503, detail="No kernels loaded in pool.")

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
                
                # Track if session provided
                if req.session_id:
                    session_tracker.track_object(req.session_id, kernel, sim, 
                                                image_id=f"img_{datetime.now().timestamp()}")

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

# ============= New Interpretability Endpoints =============

@app.post("/interpret/analyze")
async def analyze_interpretability(req: InterpretabilityRequest):
    """Analyze model interpretability without requiring retraining."""
    
    results = {}
    
    if req.analysis_type in ["projection", "all"]:
        results["projection_analysis"] = interpretability_engine.analyze_projection_space()
    
    if req.analysis_type in ["clustering", "all"]:
        results["kernel_clustering"] = interpretability_engine.analyze_kernel_clustering()
    
    if req.analysis_type in ["hierarchy", "all"]:
        results["hierarchy_analysis"] = interpretability_engine.analyze_hierarchy_graph()
    
    if req.analysis_type in ["kernels", "all"]:
        results["kernel_statistics"] = kernel_pool.get_kernel_statistics()
    
    return JSONResponse(content=results)

@app.post("/interpret/visualize_kernels")
async def visualize_kernel_space():
    """Generate visualization of kernel space."""
    clustering_data = interpretability_engine.analyze_kernel_clustering()
    
    if 'error' in clustering_data:
        raise HTTPException(status_code=400, detail=clustering_data['error'])
    
    if 'tsne_coords' in clustering_data:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # TSNE plot
        tsne_coords = np.array(clustering_data['tsne_coords'])
        ax1.scatter(tsne_coords[:, 0], tsne_coords[:, 1], alpha=0.6)
        ax1.set_title('Kernel Space (t-SNE)')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        
        # PCA plot
        if 'pca_coords' in clustering_data and len(clustering_data['pca_coords'][0]) >= 2:
            pca_coords = np.array(clustering_data['pca_coords'])
            ax2.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.6)
            ax2.set_title('Kernel Space (PCA)')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return JSONResponse(content={
            'visualization': f'data:image/png;base64,{img_base64}',
            'kernel_count': len(clustering_data.get('kernel_uuids', []))
        })
    
    return JSONResponse(content={'error': 'Insufficient data for visualization'})

@app.post("/similarity/search")
async def similarity_search(req: SimilaritySearchRequest):
    """Find most similar images from a set of candidates."""
    try:
        # Process reference image
        ref_data = base64.b64decode(req.reference_image_base64)
        ref_img = Image.open(BytesIO(ref_data)).convert("RGB")
        clip_size = clip_processor.model.config.vision_config.image_size
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
            
            # Compute similarity
            sim = F.cosine_similarity(ref_latent.unsqueeze(0), cand_latent.unsqueeze(0)).item()
            similarities.append({'index': idx, 'similarity': sim})
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return JSONResponse(content={
            'top_matches': similarities[:req.top_k],
            'all_similarities': similarities
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/track/objects")
async def track_objects(req: ObjectTrackingRequest):
    """Track objects across multiple images in a session."""
    try:
        # Process current image
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        clip_size = clip_processor.model.config.vision_config.image_size
        resized = img.resize((clip_size, clip_size))
        proc = clip_processor(images=resized, return_tensors="pt").to(device)
        vis_feat = clip_model.get_image_features(**proc).detach().squeeze(0)
        
        with torch.no_grad():
            proj_vis = projections.to_latent_space(vis_feat.unsqueeze(0).to(device)).squeeze(0)
        
        # Find matching kernels
        matches = kernel_pool.find_best_match(proj_vis.unsqueeze(0), top_k=10)
        
        # Track objects
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
        
        # Get session summary
        session = session_tracker.get_or_create_session(req.session_id)
        
        # Identify persistent objects (seen multiple times)
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
            'persistent_objects': persistent_objects[:5]  # Top 5 most persistent
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hierarchy/analyze_relationships")
async def analyze_hierarchical_relationships():
    """Analyze hierarchical relationships between objects."""
    try:
        # Compute hierarchical embeddings
        with torch.no_grad():
            hierarchical_embs = hierarchy.compute_hierarchical_embeddings(kernel_pool, device)
        
        if hierarchical_embs.numel() == 0:
            return JSONResponse(content={'error': 'No hierarchical embeddings available'})
        
        # Analyze the hierarchical structure
        hierarchy_analysis = interpretability_engine.analyze_hierarchy_graph()
        
        # Find objects with most relationships
        relationship_counts = {}
        for kernel in kernel_pool.kernels:
            rel_count = sum(len(related) for related in kernel.related_kernels.values())
            relationship_counts[kernel.uuid] = {
                'count': rel_count,
                'types': list(kernel.related_kernels.keys()),
                'label': kernel.properties.get('label', 'unknown')
            }
        
        # Sort by relationship count
        top_connected = sorted(relationship_counts.items(), 
                              key=lambda x: x[1]['count'], 
                              reverse=True)[:10]
        
        # Compute hierarchical similarity matrix
        if len(kernel_pool.kernels) > 1:
            sim_matrix = F.cosine_similarity(
                hierarchical_embs.unsqueeze(0), 
                hierarchical_embs.unsqueeze(1), 
                dim=-1
            )
            
            # Find clusters based on similarity
            threshold = 0.7
            clusters = []
            visited = set()
            
            for i in range(len(kernel_pool.kernels)):
                if i not in visited:
                    cluster = []
                    stack = [i]
                    while stack:
                        idx = stack.pop()
                        if idx not in visited:
                            visited.add(idx)
                            cluster.append(kernel_pool.kernels[idx].uuid)
                            # Find similar kernels
                            similar = (sim_matrix[idx] > threshold).nonzero().squeeze(-1)
                            for sim_idx in similar:
                                if sim_idx.item() not in visited:
                                    stack.append(sim_idx.item())
                    if cluster:
                        clusters.append(cluster)
        else:
            clusters = []
        
        return JSONResponse(content={
            'hierarchy_stats': hierarchy_analysis,
            'top_connected_objects': top_connected,
            'hierarchical_clusters': {
                'num_clusters': len(clusters),
                'cluster_sizes': [len(c) for c in clusters],
                'largest_cluster': clusters[0] if clusters else []
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

@app.post("/demo/test_pipeline")
async def test_complete_pipeline():
    """Demo endpoint to test the complete pipeline with synthetic data."""
    try:
        # Create synthetic test image (white noise)
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Convert to base64
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
        interp_req = InterpretabilityRequest(analysis_type="all")
        interp_response = await analyze_interpretability(interp_req)
        
        # Test hierarchy analysis
        hierarchy_response = await analyze_hierarchical_relationships()
        
        return JSONResponse(content={
            'pipeline_test': 'success',
            'prediction_results': {
                'num_kernels': pred_response.num_kernels,
                'num_matches': len(pred_response.matched_kernels)
            },
            'interpretability_available': list(interp_response.keys()) if isinstance(interp_response, dict) else [],
            'hierarchy_available': 'hierarchy_stats' in hierarchy_response if isinstance(hierarchy_response, dict) else False,
            'test_image_base64': img_base64  # Return for verification
        })
        
    except Exception as e:
        return JSONResponse(content={
            'pipeline_test': 'failed',
            'error': str(e)
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

# ============= Utility Endpoints for Visualization =============

@app.post("/visualize/attention_map")
async def visualize_attention(req: PredictRequest):
    """Visualize pseudo-attention between image and text."""
    try:
        # Process inputs
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        
        clip_size = clip_processor.model.config.vision_config.image_size
        resized = img.resize((clip_size, clip_size))
        proc = clip_processor(images=resized, return_tensors="pt").to(device)
        vis_feat = clip_model.get_image_features(**proc).detach()
        
        text_desc = req.text_description if req.text_description else "an object"
        tproc = clip_processor(text=[text_desc], return_tensors="pt", padding=True, truncation=True).to(device)
        text_feat = clip_model.get_text_features(**tproc).detach()
        
        # Compute attention maps
        attention_maps = interpretability_engine.compute_attention_maps(vis_feat, text_feat)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(attention_maps.squeeze(), cmap='hot', interpolation='nearest')
        ax.set_title(f'Attention Map: "{text_desc}"')
        plt.colorbar(im, ax=ax)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return JSONResponse(content={
            'attention_visualization': f'data:image/png;base64,{img_base64}',
            'max_attention': float(attention_maps.max()),
            'min_attention': float(attention_maps.min()),
            'mean_attention': float(attention_maps.mean())
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    print(f"Starting enhanced server with interpretability...")
    print(f"Device: {device}")
    print(f"Checkpoint loaded: {_loaded_ckpt_path}")
    print(f"Kernels in pool: {len(kernel_pool.kernels)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)