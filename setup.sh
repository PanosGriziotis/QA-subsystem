#!/bin/bash

# Get the absolute path of the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if the virtual environment directory exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment already exists. Recreating it..."
    rm -rf "$SCRIPT_DIR/venv"
fi

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the dependencies
pip install  --no-cache-dir -r requirements.txt

# Install torch to match available GPU drivers
#pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Start the Elasticsearch service using a Docker container
python3 document_store/initialize_document_store.py --launch

# Run the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000
