import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import sys
import json
import logging
import re # For entity extraction in the memory system's example

from flask import Flask, request, jsonify
from flask_cors import CORS # To allow cross-origin requests from the frontend

# Configure logging for the backend
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BackendAgent")

# Add the directory containing memory_system.py to the Python path
# Assuming memory_system.py is in the same directory as this script.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_system import ContextManager # Import the memory system

# --- Configuration ---
MODEL_DIR = "phi2_model" # This directory was created by the previous script

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Global Model and Context Manager Instances ---
tokenizer = None
model = None
context_manager = None

def load_phi2_model():
    """Loads the Phi-2 model and tokenizer."""
    global tokenizer, model
    try:
        logger.info(f"Loading Phi-2 model from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16) # Use bfloat16 for efficiency
        model.eval() # Set model to evaluation mode
        logger.info("Phi-2 model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Phi-2 model: {e}", exc_info=True)
        sys.exit(1)

def get_embedding(text: str) -> np.ndarray:
    """
    Generates a sentence embedding using mean pooling of Phi-2's last layer hidden states.
    """
    if model is None or tokenizer is None:
        logger.error("Phi-2 model not loaded. Cannot generate embedding.")
        return np.zeros(2560, dtype=np.float32) # Return a zero vector or handle error appropriately

    if not text.strip():
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden_states = outputs.hidden_states[-1]
    attention_mask = inputs.attention_mask

    sum_embeddings = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask.unsqueeze(-1)

    return mean_embeddings[0].cpu().numpy()

def generate_phi2_response(prompt_text: str, max_new_tokens: int = 200) -> str:
    """
    Generates text using the Phi-2 model based on a prompt.
    """
    if model is None or tokenizer is None:
        logger.error("Phi-2 model not loaded. Cannot generate response.")
        return "Error: Agent not ready."

    inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1, # Simple greedy decoding for speed
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id # Important for generation
        )

    # Decode the generated tokens, skipping the input prompt
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    if generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):].strip()
    return generated_text.strip()

@app.before_first_request
def initialize_agent():
    """Initializes the Phi-2 model and ContextManager on first request."""
    global context_manager
    logger.info("Initializing Agent (loading models and memory)...")
    load_phi2_model()
    context_manager = ContextManager()
    logger.info("Agent initialization complete.")

def _extract_entities_from_text(text: str) -> Set[str]:
    """
    Simple entity extraction for demonstration.
    Looks for capitalized words or phrases.
    """
    entities = set()
    matches = re.findall(r'\b[A-Z][a-z0-9]*(?:\s+[A-Z][a-z0-9]*)*\b', text)
    for match in matches:
        if len(match) > 1 or (len(match) == 1 and match.isupper() and match.isascii()):
            if not match.isupper() or (' ' in match): # Exclude single-word ALL_CAPS unless multi-word
                entities.add(match.lower())
    return entities

@app.route('/agent_query', methods=['POST'])
def agent_query():
    """
    Endpoint for the frontend to send user prompts.
    Processes the prompt, integrates with memory, and generates a response.
    """
    data = request.json
    user_prompt = data.get('user_prompt', '')

    if not user_prompt:
        return jsonify({"agent_response": "Please provide a query."}), 400

    logger.info(f"Received query: '{user_prompt}'")

    # 1. Generate embedding for the user prompt
    prompt_embedding = get_embedding(user_prompt)

    # 2. Extract entities from the user prompt (simple heuristic)
    entities = _extract_entities_from_text(user_prompt)

    # 3. Process the input with the ContextManager
    # This will update STM, LTM, and retrieve relevant context
    relevant_context = context_manager.process_input(
        input_text=user_prompt,
        embedding=prompt_embedding,
        entities=entities,
        emotional_valence=0.0 # Placeholder, could be from a sentiment analysis model
    )

    # 4. Construct a prompt for Phi-2 based on the user query and retrieved context
    context_str_parts = []
    if relevant_context["short_term_memories"]:
        stm_texts = [m["content"].get("text", "") for m in relevant_context["short_term_memories"]]
        context_str_parts.append("Short-Term Memory (Recent/Relevant):")
        for text in stm_texts:
            context_str_parts.append(f"- {text}")

    if relevant_context["long_term_memories"]:
        ltm_texts = [m["content"].get("text", "") for m in relevant_context["long_term_memories"]]
        if context_str_parts: context_str_parts.append("") # Add a newline if STM exists
        context_str_parts.append("Long-Term Memory (Deeper Knowledge):")
        for text in ltm_texts:
            context_str_parts.append(f"- {text}")
    
    if relevant_context["active_context_metadata"]["active_entities"]:
        if context_str_parts: context_str_parts.append("")
        context_str_parts.append(f"Current Active Entities: {', '.join(relevant_context['active_context_metadata']['active_entities'])}")

    context_prompt_section = "\n".join(context_str_parts)
    if context_prompt_section:
        context_prompt_section = "\n\nContext from Agent's Memory:\n" + context_prompt_section + "\n"

    # Define the core instruction prompt for Phi-2
    phi2_prompt = f"""You are an Autonomous Code Agent. Your goal is to be helpful, precise, and concise.
Based on the following context from your memory and the user's query, provide a clear and actionable response.
Do not repeat the user's query. Just provide the answer.

{context_prompt_section}

User Query: {user_prompt}

Agent Response:"""

    logger.debug(f"Phi-2 Prompt:\n{phi2_prompt}")

    # 5. Generate response using Phi-2
    agent_response = generate_phi2_response(phi2_prompt)
    logger.info(f"Generated response: '{agent_response[:100]}...'")

    return jsonify({"agent_response": agent_response})

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "message": "Agent backend is running.", "current_time": time.time()})

@app.route('/memory_stats', methods=['GET'])
def memory_stats():
    """Endpoint to get memory statistics."""
    if context_manager:
        stats = context_manager.get_stats()
        return jsonify(stats)
    return jsonify({"error": "Context Manager not initialized."}), 500

if __name__ == '__main__':
    # Start the Flask app. This implicitly calls before_first_request.
    # Use a separate thread for the Flask development server if needed,
    # but for simple execution, app.run() is sufficient.
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
