#!/bin/bash
echo "Setting up Violin Posture Tutor..."

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install opencv-python numpy mediapipe

echo ""
echo "Setup complete! Starting application..."
python app.py