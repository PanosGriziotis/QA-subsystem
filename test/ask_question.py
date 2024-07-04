import argparse
import requests

def main():
    parser = argparse.ArgumentParser(description="Send a query to the server and retrieve answers.")
    parser.add_argument('--rag', action='store_true', help='Use RAG pipeline to answer query.')
    parser.add_argument('--ex', action='store_true', help='Use extractive qa pipeline to answer query.')
    parser.add_argument('--query', type=str, required=True, help='The query to send to the server.')
    args = parser.parse_args()

    if not args.rag and not args.ex:
        parser.error('No action requested, add --rag or --ex')

    query = args.query
    request_body = {
        "query": query,
        "params": {
            "Retriever": {"top_k": 10},
            "Ranker": {"top_k": 10},
        }
    }

    endpoint = "rag-query" if args.rag else "extractive-query" if args.ex else None
    url = f"http://127.0.0.1:8000/{endpoint}"

    try:
        r = requests.post(url=url, json=request_body)
        r.raise_for_status()  # Raise an HTTPError for bad responses
        json_response = r.json()
        print(json_response["answers"])
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except KeyError:
        print("Unexpected response structure:", json_response)

if __name__ == "__main__":
    main()
