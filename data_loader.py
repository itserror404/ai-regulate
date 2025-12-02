from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pickle


file_paths = ["ROADMAP-FOR-AI-POLICY-RESEARCH.txt", "ai-regulation-china-eu-us.txt"]
documents = []

for file_path in file_paths:
    loader = TextLoader(file_path)
    documents.extend(loader.load())


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)


print(f"Loaded {len(split_docs)} document chunks.")


with open("split_docs.txt", "w", encoding="utf-8") as f:
    for doc in split_docs:
        f.write(doc.page_content + "\n\n")


print("Saved split documents.")