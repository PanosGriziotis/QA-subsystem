import glob
import requests


def test_application():

    # test status endpoint
    r = requests.get(url="http://127.0.0.1:8000/ready")
    assert r.status_code == 200
    assert r.text == "true"

    # test file-upload endpoint (indexing)
    for txt_file in glob.glob("example_data/*.txt"):
        with open(txt_file, "rb") as f:
            r = requests.post(url="http://127.0.0.1:8000/file-upload", files={"files": f})

            json_response = r.json()
            assert len (json_response["documents"]) > 0
           
        
    # test query endpoints
    query = "Πώς μεταδίδεται ο covid-19;"

    for endpoint in ["extractive-query", "rag-query"]:

        r = requests.get(url=f"http://127.0.0.1:8000/{endpoint}", params={"query": query})
        json_response = r.json()
        print(json_response)
        assert "answers" in json_response

        #answer = json_response["answers"][0]
        assert len(json_response["documents"]) > 0
        assert json_response["query"] == query

if __name__ == "__main__":

    test_application()