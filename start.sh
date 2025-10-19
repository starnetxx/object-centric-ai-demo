#!/bin/bash
echo "Starting CORE Object-Centric AI Server..."
echo "Python version:"
python3 --version
echo "Installing dependencies..."
pip3 install -r requirements.txt
echo "Starting server..."
python3 enhanced_server.py
