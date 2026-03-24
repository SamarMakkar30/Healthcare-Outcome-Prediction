#!/bin/bash
# Healthcare Prediction System - Quick Setup Script for Linux/Mac
# Run this file to set up and start the project

echo "========================================"
echo "Healthcare Prediction System Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Setup complete!"
echo ""
echo "========================================"
echo "Ready to run!"
echo "========================================"
echo ""
echo "Choose an option:"
echo "  1. Full setup (Generate data + Train models + Start web app)"
echo "  2. Quick start (Start web app only - requires trained models)"
echo ""
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Running full setup..."
    python run_project.py
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Starting web application..."
    cd web
    python app.py
else
    echo "Invalid choice. Please run the script again."
fi
