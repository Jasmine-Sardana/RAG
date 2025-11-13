from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("GenAi Notes.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text = text_splitter.split_documents(documents)

#loading the embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
print("Embedding Model Loaded....")

url = "http://localhost:6333"
collection_name = "gpt_db"

qdrant = Qdrant.from_documents(
    text,
    embeddings,
    url =url,
    prefer_grpc = False,
    collection_name=collection_name
)

print("Documents ingested successfully into Qdrant vector store.")

