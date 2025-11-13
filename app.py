from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

#loading the embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
print("Embedding Model Loaded....")

url = "http://localhost:6333"
collection_name = "gpt_db"

client = QdrantClient(
    url=url,
    prefer_grpc=False
)

print(client)
print("####")

db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings
)

print(db)
print("####")

query = "What are core generative ai models?"

docs= db.similarity_search_with_score(query=query, k=5)

for i in docs:
    doc, score = i
    print({"Score": score, "Content": doc.page_content, "Metadata": doc.metadata})