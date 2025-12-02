from langchain_qdrant import Qdrant                     # Qdrant vector store for remote Qdrant[web:21][web:96]
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load and split PDF
loader = PyPDFLoader("GenAi Notes.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# 2. Embeddings (BGE)
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
print("Embedding Model Loaded....")

# 3. Ingest into Qdrant (remote HTTP)
url = "http://localhost:6333"
collection_name = "gpt_db"

qdrant = Qdrant.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    url=url,
    prefer_grpc=False,
    collection_name=collection_name,
)

print("Documents ingested successfully into Qdrant vector store.")
