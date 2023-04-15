# rest api  test

import requests
import sys


if __name__ == '__main__':

    q = sys.argv[1]

    url = "http://localhost:8000/query"
    payload = {"query": q}
    headers = {
                'Content-Type': 'application/json'
            }
    response = requests.request("POST", url, headers=headers, json=payload).json()

    if response["answers"]:
        answer = response["answers"][0]["answer"]
    else:
        answer = "No Answer Found!"

    print (answer)