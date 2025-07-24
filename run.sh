#!/bin/bash

MODEL_DIR="phi2_model"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

# Check if the model is already downloaded.
# This checks if the directory exists AND if it's not empty.
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Model not found or directory is empty. Running model download script..."
    python download_model.py
    if [ $? -ne 0 ]; then
        echo "Model download failed. Please check the error messages above. Exiting."
        exit 1
    fi
else
    echo "Model directory exists and is not empty. Skipping model download."
fi

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment '$VENV_DIR' not found. It should have been created by download_model.py. Exiting."
    exit 1
fi

# Install requirements again, just in case some were missed or added later
echo "Ensuring all Python dependencies are installed from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"

echo "Starting Autonomous Code Agent backend..."
# The Flask app will listen on 0.0.0.0:5000 by default in app.run()
python backend_agent.py

echo "Backend stopped."
