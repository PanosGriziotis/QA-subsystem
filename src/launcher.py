import argparse
import subprocess
import os


def launch_app():
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

def ingest_data():
    subprocess.run(["python3", "external_data/ingest_data_to_doc_store.py"])

def main():
    parser = argparse.ArgumentParser(description="A script to launch the rest api server or ingest data to document store")
    parser.add_argument('--launch_app', action='store_true', help='Launch the server using uvicorn')
    parser.add_argument('--ingest_data', action='store_true', help='Ingest data to the document store')

    args = parser.parse_args()

    if args.launch_app:
        launch_app()
    elif args.ingest_data:
        ingest_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()