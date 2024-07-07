import os
from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import nltk

usePunkt = True
# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    usePunkt = False
    # nltk.download('punkt')


class DocumentRetriever:
    def __init__(self, data_dir: str, embedding_model: str = "text-embedding-ada-002"):
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.vector_store = None

    def load_documents_with_nltk(self) -> List[Dict]:
        loader = DirectoryLoader(self.data_dir, glob="**/*.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        return split_docs

    def load_documents_simple(self) -> List[Dict]:
        documents = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        documents.append(
                            Document(page_content=text, metadata={"source": file_path}))

        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    def create_vector_store(self, documents: List[Dict]):
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = Chroma.from_documents(documents, embeddings)

    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[Dict]:
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call create_vector_store() first.")
        return self.vector_store.similarity_search(query, k=k)

    def setup(self):
        documents = self.load_documents_with_nltk if usePunkt else self.load_documents_simple()
        self.create_vector_store(documents)
