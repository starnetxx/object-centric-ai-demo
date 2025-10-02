"""
inference/server.py

FastAPI server that loads checkpoint with kernel pool and models and exposes /predict.
Keep this lightweight; for production consider batching and async GPU management.
"""

import os
import base64
from io import BytesIO
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as T

# from model.projection import ProjectionEngine
# from model.kernels import KernelPool, ObjectKernel
# from model.hierarchy import HierarchyEngine
from transformers import CLIPProcessor, CLIPModel
import time
import uuid
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean
# Fallback implementation for scatter_mean
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
    count = count.clamp(min=1)
    result = result / count.unsqueeze(1).float()
    return result
import torch_geometric.nn as geo_nn

# Re-define necessary classes (ensure these match the definitions used during training)
class ProjectionEngine(nn.Module):
    """
    Handles projection between latent space and CLIP space.
    """
    def __init__(self, latent_dim=256, clip_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.projection_matrix = nn.Parameter(torch.randn(latent_dim, clip_dim) * 0.01)

    def to_clip_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects latent -> CLIP space.
        """
        return x @ self.projection_matrix

    def to_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects CLIP -> latent space by computing pseudo-inverse on the fly.
        """
        pinv = torch.linalg.pinv(self.projection_matrix)
        return x @ pinv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects CLIP -> latent space by default.
        """
        return self.to_latent_space(x)

class ObjectKernel(nn.Module):
    """
    Persistent object identity with GRU-based updates and metadata.
    """
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

    def forward(self):
        return self.state

    def update_state(self, new_emb: torch.Tensor):
        if new_emb.dim() == 2 and new_emb.shape[0] == 1:
            new_emb = new_emb.squeeze(0)
        if new_emb.numel() == 0:
            return
        with torch.no_grad():
            updated = self.updater(new_emb.to(self.state.device), self.state)
            self.state.copy_(updated.detach())
            self.last_updated = time.time()

    def add_relationship(self, relation_type: str, other_uuid: str):
        if relation_type not in self.related_kernels:
            self.related_kernels[relation_type] = []
        if other_uuid not in self.related_kernels[relation_type]:
            self.related_kernels[relation_type].append(other_uuid)

    def get_embedding(self):
        return self.state.detach()

class KernelPool:
    """
    Manage a pool of ObjectKernel instances with aging/pruning policies.
    """
    def __init__(self, max_kernels=500, min_age_seconds=60.0, prune_to=400):
        self.kernels = []  # list of ObjectKernel
        self.max_kernels = max_kernels
        self.min_age_seconds = min_age_seconds
        self.prune_to = prune_to

    def add_kernel(self, k: ObjectKernel):
        self.kernels.append(k)
        self._prune_if_needed()

    def _prune_if_needed(self):
        if len(self.kernels) <= self.max_kernels:
            return
        self.kernels.sort(key=lambda x: x.last_updated)
        while len(self.kernels) > self.prune_to:
            removed = self.kernels.pop(0)

    def find_best_match(self, query_emb: torch.Tensor, top_k=1):
        if not self.kernels:
            return []
        emb_matrix = torch.stack([k.get_embedding().to(query_emb.device) for k in self.kernels], dim=0)
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        sim = F.cosine_similarity(query_emb.unsqueeze(1), emb_matrix.unsqueeze(0), dim=-1)
        topk_vals, topk_idx = torch.topk(sim, k=min(top_k, emb_matrix.size(0)), dim=1)
        results = []
        for b in range(topk_idx.size(0)):
            results.append([(self.kernels[idx.item()], topk_vals[b, i].item()) for i, idx in enumerate(topk_idx[b])])
        return results

    def to_state_dict(self):
        return [ {
            "uuid": k.uuid, "state": k.state.detach().cpu(), "properties": k.properties,
            "related_kernels": k.related_kernels, "last_updated": k.last_updated
        } for k in self.kernels ]

    def load_from_state_list(self, state_list, device='cpu'):
        self.kernels = []
        for s in state_list:
            # Ensure latent_dim is correctly inferred from the state shape
            k = ObjectKernel(latent_dim=s["state"].numel())
            k.state.data.copy_(s["state"].to(device))
            k.properties = s.get("properties", {})
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


CHECKPOINT_DIR = "checkpoints"
# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

app = FastAPI(title="Object-Centric AI API (Refactored)")

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

# Globals (loaded at startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate models with dimensions matching the training script's latent_dim and gnn_out
latent_dim = 128
clip_dim = 512
projections = ProjectionEngine(latent_dim=latent_dim, clip_dim=clip_dim).to(device)
hierarchy = HierarchyEngine(latent_dim=latent_dim, gnn_out=256).to(device)
kernel_pool = KernelPool(max_kernels=500)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Load specific checkpoint
def load_checkpoint_for_inference(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f"[server] checkpoint not found: {checkpoint_path}")
        return None
    try:
        state = torch.load(checkpoint_path, map_location=device)
        # load projection and gnn
        projections.load_state_dict(state["projections"])
        if state.get("hierarchy_gnn") is not None:
            hierarchy.gnn.load_state_dict(state["hierarchy_gnn"])
        # restore kernel pool
        kp_state = state.get("kernel_pool", [])
        kernel_pool.load_from_state_list(kp_state, device=device)
        print(f"[server] loaded checkpoint {checkpoint_path} with {len(kernel_pool.kernels)} kernels")
        return checkpoint_path
    except Exception as e:
        print(f"[server] failed to load checkpoint {checkpoint_path}: {e}")
        print(f"[server] continuing with untrained models...")
        return None

# Specify the path to the checkpoint you want to load
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "coco_ckpt_epoch_4.pth") # Or "coco_ckpt_epoch_2.pth"
_loaded_ckpt_path = load_checkpoint_for_inference(CHECKPOINT_PATH, device=device)

class PredictRequest(BaseModel):
    image_base64: str

class PredictResponse(BaseModel):
    num_kernels: int
    matched_kernel_ids: List[str]
    matched_scores: List[float]

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # Continue even if checkpoint loading failed - use untrained models
    if _loaded_ckpt_path is None:
        print("[server] Using untrained models for prediction...")

    try:
        data = base64.b64decode(req.image_base64)
        img = Image.open(BytesIO(data)).convert("RGB")
        # img_t = T.ToTensor()(img).unsqueeze(0).to(device) # Not used in simplified demo inference

        # Simple object detection fallback: we skip detectron for demo; crop whole image as single object
        clip_size = 224  # Standard CLIP input size
        resized = img.resize((clip_size, clip_size))
        proc = clip_processor(images=resized, return_tensors="pt").to(device)
        vis_feat = clip_model.get_image_features(**proc).detach().squeeze(0)  # [512]
        # text: we use open set of common categories; for demo use "an object"
        text_desc = "a photo of an object"
        tproc = clip_processor(text=[text_desc], return_tensors="pt", padding=True, truncation=True).to(device)
        text_feat = clip_model.get_text_features(**tproc).detach().squeeze(0)

        # project to latent
        with torch.no_grad():
            # Project vision feature to latent space
            proj_vis = projections.to_latent_space(vis_feat.unsqueeze(0).to(device)).squeeze(0)
            # Project text feature to latent space
            proj_text = projections.to_latent_space(text_feat.unsqueeze(0).to(device)).squeeze(0)


        # Match against kernel_pool using the projected features
        if len(kernel_pool.kernels) == 0:
            # Create a dummy kernel for demo purposes
            dummy_kernel = ObjectKernel(latent_dim=latent_dim)
            dummy_kernel.update_state((proj_vis + proj_text) / 2.0)
            kernel_pool.add_kernel(dummy_kernel)
            print(f"[server] Created dummy kernel for demo: {dummy_kernel.uuid}")

        # For demo, match against the average of projected vision and text features
        query_emb = (proj_vis + proj_text) / 2.0

        matches = kernel_pool.find_best_match(query_emb.unsqueeze(0), top_k=5) # Unsqueeze for batch dim
        matched_ids = [k.uuid for k, sim in matches[0]] if matches and matches[0] else []
        matched_scores = [sim for k, sim in matches[0]] if matches and matches[0] else []


        return PredictResponse(num_kernels=len(kernel_pool.kernels), matched_kernel_ids=matched_ids, matched_scores=matched_scores)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))