mport os
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
