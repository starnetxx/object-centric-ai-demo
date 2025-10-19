#!/bin/bash
echo "Starting CORE Object-Centric AI Server..."
echo "Python version:"
python --version
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting server..."
python enhanced_server.py
