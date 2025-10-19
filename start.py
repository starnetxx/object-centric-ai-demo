#!/usr/bin/env python3
"""
Simple startup script for Railway
"""
import subprocess
import sys
import os

# Install dependencies
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Start the server
print("Starting server...")
os.system("python3 enhanced_server.py")
