from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as P

import os
from langchain_community.document_loaders import TextLoader

import pickle

os.environ['PINECONE_API_KEY'] = ""

index_name = "hello"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

loader = TextLoader("split_docs.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(
        docs,
        index_name=index_name,
        embedding=embedding_model
    )