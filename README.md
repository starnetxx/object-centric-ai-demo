# Object-Centric AI Demo

A demonstration of object-centric AI using CLIP, Graph Neural Networks, and persistent object kernels for image understanding and object recognition.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (tested on Windows)
- At least 4GB RAM
- Internet connection for downloading models

### Installation

1. **Clone or download the project**
   ```bash
   cd C:\Users\DEEP\Music\Firstdemo\Firstdemo
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision
   pip install transformers datasets matplotlib scikit-learn wandb fastapi uvicorn pydantic pillow
   pip install torch-geometric
   ```

4. **Start the server**
   ```bash
   python start_server.py
   ```

5. **Open the web interface**
   - Open `frontend/index.html` in your web browser
   - Or visit `http://localhost:8000/docs` for API documentation

## 📁 Project Structure

```
Firstdemo/
├── backend/
│   ├── server.py              # FastAPI server with AI models
│   ├── requirements.txt       # Python dependencies
│   └── checkpoints/           # Model checkpoints (auto-created)
├── frontend/
│   └── index.html             # Web interface
├── start_server.py            # Server startup script
├── create_dummy_checkpoint.py # Creates dummy model checkpoint
└── README.md                  # This file
```

## 🔧 How It Works

### Backend (FastAPI Server)
- **CLIP Integration**: Uses OpenAI's CLIP model for image and text understanding
- **Object Kernels**: Maintains persistent object representations with GRU-based updates
- **Graph Neural Networks**: Processes hierarchical object relationships
- **Projection Engine**: Maps between CLIP space and latent space
- **REST API**: Provides `/predict` endpoint for image analysis

### Frontend (HTML Interface)
- **Image Upload**: Simple drag-and-drop interface
- **Base64 Encoding**: Converts images for API transmission
- **Results Display**: Shows matched object kernels and similarity scores

## 🎯 Features

- **Object Detection**: Analyzes uploaded images using CLIP
- **Persistent Memory**: Maintains object kernels across sessions
- **Similarity Matching**: Finds similar objects in the kernel pool
- **Real-time Processing**: Fast inference with pre-trained models
- **Web Interface**: Easy-to-use HTML frontend

## 🚀 Usage

### Starting the Server
```bash
# Local development
python simple_ai_server.py

# Or using the startup script (for complex server)
python start_server.py
```

### Deployment
This project is ready for deployment on:
- **Backend**: Railway, Render, or Heroku
- **Frontend**: Netlify, Vercel, or GitHub Pages

### Using the Web Interface
1. Open `frontend/index.html` in your browser
2. Click "Choose File" to select an image
3. Click "Predict" to analyze the image
4. View the results showing matched object kernels

### API Usage
```bash
# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "your_base64_encoded_image"}'
```

## 🔍 Technical Details

### Models Used
- **CLIP**: `openai/clip-vit-base-patch32` for vision and text understanding
- **PyTorch**: For neural network operations
- **PyTorch Geometric**: For graph neural network operations
- **FastAPI**: For REST API server

### Architecture
1. **Image Processing**: CLIP processes uploaded images
2. **Feature Extraction**: Vision and text features are extracted
3. **Latent Projection**: Features are projected to latent space
4. **Kernel Matching**: Query is matched against object kernels
5. **Response**: Similarity scores and kernel IDs are returned

## 🛠️ Troubleshooting

### Common Issues

1. **Server won't start**
   - Ensure all dependencies are installed
   - Check if port 8000 is available
   - Verify Python virtual environment is activated

2. **Import errors**
   - Reinstall PyTorch: `pip install torch torchvision`
   - Install PyTorch Geometric: `pip install torch-geometric`

3. **Checkpoint loading errors**
   - The server will continue with untrained models if checkpoints fail to load
   - This is normal for demo purposes

4. **CORS issues**
   - The frontend should work with the backend on localhost
   - For production, configure CORS settings in FastAPI

### Performance Notes
- First image processing may take longer due to model loading
- Subsequent requests will be faster
- GPU acceleration is supported if CUDA is available

## 📊 API Endpoints

### POST /predict
Analyzes an uploaded image and returns object kernel matches.

**Request Body:**
```json
{
  "image_base64": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "num_kernels": 1,
  "matched_kernel_ids": ["uuid-string"],
  "matched_scores": [0.95]
}
```

### GET /docs
Interactive API documentation (Swagger UI)

## 🔮 Future Enhancements

- [ ] Add more sophisticated object detection
- [ ] Implement object relationship learning
- [ ] Add support for video input
- [ ] Improve kernel pool management
- [ ] Add user authentication
- [ ] Implement model training interface

## 📝 License

This project is for demonstration purposes. Please check individual model licenses (CLIP, PyTorch, etc.) for commercial use.

## 🤝 Contributing

This is a demo project. For questions or issues, please check the troubleshooting section above.

---

**Note**: This demo uses dummy checkpoints and simplified object detection. For production use, proper model training and checkpoint management would be required.
