# CORE - Object-Centric AI Frontend

This is the frontend for the CORE Object-Centric AI application, designed to be deployed on Netlify.

## 🚀 Deployment to Netlify

### Option 1: Deploy from GitHub (Recommended)

1. **Connect to GitHub**:
   - Go to [Netlify](https://netlify.com)
   - Click "New site from Git"
   - Connect your GitHub account
   - Select your repository: `starnetxx/object-centric-ai-demo`

2. **Configure Build Settings**:
   - **Base directory**: `frontend`
   - **Build command**: `echo 'No build step required'`
   - **Publish directory**: `frontend`

3. **Deploy**:
   - Click "Deploy site"
   - Netlify will automatically deploy your frontend

### Option 2: Manual Deploy

1. **Build the frontend**:
   ```bash
   cd frontend
   # No build step required - it's static HTML
   ```

2. **Deploy to Netlify**:
   - Drag and drop the `frontend` folder to Netlify
   - Or use Netlify CLI: `netlify deploy --dir=frontend`

## 🔧 Configuration

The frontend automatically detects the environment:
- **Local**: Uses `http://localhost:8000`
- **Netlify**: Uses your Railway backend URL
- **Production**: Uses your Railway backend URL

## 📁 Files

- `index.html` - Main application
- `config.js` - Environment configuration
- `CORE 01.png` - Logo
- `favicon.png` - Favicon
- `_redirects` - Netlify redirects
- `netlify.toml` - Netlify configuration

## 🌐 Live URLs

- **Frontend (Netlify)**: Your Netlify URL
- **Backend (Railway)**: `https://web-production-e7ea7.up.railway.app`

## 🔄 Auto-Deploy

Once connected to GitHub, Netlify will automatically redeploy when you push changes to the `main` branch.
