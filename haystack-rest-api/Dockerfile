FROM python:3.8.10

WORKDIR /code

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="${PYTHONPATH}:/code/src:/code"

COPY ./requirements.txt /code/requirements.txt

RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt



COPY ./src /code/src
COPY ./ingest_example_data.py /code/ingest_example_data.py

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]