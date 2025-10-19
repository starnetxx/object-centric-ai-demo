#!/usr/bin/env python
"""
Railway startup script for CORE Object-Centric AI Server
"""
import subprocess
import sys
import os

def main():
    print("🚀 Starting CORE Object-Centric AI Server...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)
    
    # Start the enhanced server
    print("🌐 Starting enhanced server...")
    try:
        # Import and run the enhanced server
        from enhanced_server import app
        import uvicorn
        
        port = int(os.environ.get("PORT", 8000))
        print(f"Server starting on port {port}")
        
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
