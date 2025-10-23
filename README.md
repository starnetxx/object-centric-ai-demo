---
title: CORE Object-Centric AI
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# CORE Object-Centric AI

A powerful object-centric AI system with persistent object kernels, CLIP-based understanding, and hierarchical relationships.

## Features

- **Real-time Object Detection**: Using CLIP model for image understanding
- **Similarity Search**: Find similar objects across images
- **Object Tracking**: Persistent object kernels across sessions
- **Interpretability**: Model analysis and visualization
- **Hierarchical Analysis**: Object relationship understanding

## API Endpoints

- `POST /predict` - Object detection and analysis
- `GET /health` - Health check
- `GET /docs` - API documentation

## Usage

Upload images and get AI-powered object detection and analysis results.

## Technology Stack

- FastAPI for the backend
- PyTorch and Transformers for AI models
- CLIP for vision-language understanding
- Scikit-learn for analysis