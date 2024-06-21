FROM deepset/haystack-gpu


WORKDIR /code

# Install required packages and clean up
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
#ENV CUDA_VERSION=11.4
#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="${PYTHONPATH}:/code/src:/code"

# Copy the requirements file and install dependencies
COPY ./req.txt /code/req.txt

# Install Python packages
RUN pip install --no-cache-dir --upgrade -r /code/req.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cu118 \
    && pip install accelerate bitsandbytes xformers

# Copy the source code
COPY ./src /code/src
# COPY ./ingest_example_data.py /code

# (Optional) Specify an entry point or command to run your application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]