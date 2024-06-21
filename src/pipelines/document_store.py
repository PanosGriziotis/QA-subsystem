from haystack.document_stores import ElasticsearchDocumentStore

HOST = "localhost"
PORT = 9200

document_store = ElasticsearchDocumentStore(
    host=HOST,
    port=PORT,
    username="",
    password="",
    index="test_api",
    duplicate_documents="overwrite"
)
