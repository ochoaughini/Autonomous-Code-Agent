#!/bin/bash

# Ensure Python virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Error: Virtual environment 'venv' not found. Please run the model download script first to set it up."
        exit 1
    fi
fi

# Install dependencies if not already installed
pip install -r requirements.txt

echo "Starting Autonomous Code Agent backend..."
python backend_agent.py

echo "Backend stopped."
