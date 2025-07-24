The provided code implements a sophisticated memory system for an AI agent, integrating short-term and long-term memory components with a Flask backend and a React frontend.

To make this project runnable, you need to:

Download the Phi-2 model: The backend_agent.py relies on a locally stored Phi-2 model. This is not included in the provided files.
Set up a Python virtual environment: Best practice for managing dependencies.
Install dependencies: requirements.txt lists them.
Run the backend server: The run.sh script is intended for this.
I will provide the necessary download_model.py script and update run.sh to ensure a smooth setup and execution.

1. Create download_model.py

This script will set up the virtual environment, install the core dependencies, and download the Phi-2 model.

download_model.py

import os
import subprocess
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelDownloader")

MODEL_NAME = "microsoft/phi-2"
MODEL_DIR = "phi2_model"
VENV_DIR = "venv"

def ensure_venv():
    """Ensures a Python virtual environment exists."""
    if not os.path.exists(VENV_DIR):
        logger.info(f"Creating virtual environment at ./{VENV_DIR}...")
        try:
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            logger.info("Virtual environment created.")
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        logger.info(f"Virtual environment at ./{VENV_DIR} already exists.")

def install_packages():
    """Installs core packages into the virtual environment."""
    # Determine the correct pip path for the virtual environment
    pip_path = os.path.join(VENV_DIR, "bin", "pip")
    if os.name == 'nt':  # For Windows
        pip_path = os.path.join(VENV_DIR, "Scripts", "pip.exe")

    logger.info("Installing necessary packages into virtual environment...")
    try:
        # Install transformers, torch, numpy, and pyyaml
        subprocess.run([pip_path, "install", "torch", "transformers", "numpy", "pyyaml"], check=True)
        logger.info("Core packages installed.")
    except Exception as e:
        logger.error(f"Failed to install packages: {e}")
        sys.exit(1)

def download_model():
    """Downloads the Phi-2 model and tokenizer to MODEL_DIR."""
    if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        logger.info(f"Model already downloaded to ./{MODEL_DIR}. Skipping download.")
        return

    logger.info(f"Downloading {MODEL_NAME} to ./{MODEL_DIR}...")
    try:
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Save them to the specified directory
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
        logger.info("Model and tokenizer downloaded and saved successfully.")
    except Exception as e:
        logger.error(f"Error downloading model: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    ensure_venv()
    install_packages()
    download_model()
    logger.info("Model download process completed. You can now run the backend with ./run.sh")
2. Update run.sh

This updated script will first check if the model is downloaded. If not, it will run download_model.py. Then it will activate the virtual environment and start the backend_agent.py.

run.sh

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
3. Files in project directory

Ensure your project directory has the following structure:

your_project_directory/
├── memory_system.py
├── backend_agent.py
├── index.html
├── requirements.txt
├── run.sh
└── download_model.py  <-- NEW FILE
4. How to Run

Open your terminal or command prompt.

Navigate to your project directory (e.g., cd path/to/your_project_directory).

Make run.sh executable (if on Linux/macOS):

chmod +x run.sh
Run the setup and backend:

./run.sh
The first time you run this, it will:

Create a venv (virtual environment) directory.
Install torch, transformers, numpy, pyyaml into the venv.
Download the microsoft/phi-2 model into a phi2_model directory. This step can take a while depending on your internet speed.
Install other dependencies from requirements.txt.
Start the Flask backend server.
Access the frontend: Once the backend server starts (you'll see Flask logs in your terminal, typically * Running on http://0.0.0.0:5000), open your web browser and navigate to the index.html file. You can simply open it directly from your file system (e.g., file:///path/to/your_project_directory/index.html).

You should now see the "Autonomous Code Agent" frontend. Type your query into the text area and click "Engage Agent". The backend will process your query, utilize the memory system, generate an embedding, and formulate a response using the Phi-2 model.
