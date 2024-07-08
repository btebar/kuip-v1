from doc_retriever import DocumentRetriever
from generation import RAGGenerator

import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the documents directory
# Assuming it's in a 'documents' folder at the project root
documents_path = os.path.join(current_dir, '.', 'test_data')

# Set up the retriever
retriever = DocumentRetriever(documents_path)
retriever.setup()

# Set up the generator
generator = RAGGenerator(retriever)
generator.setup_qa_chain()


def get_rag_answer(question: str) -> str:
    return generator.generate_answer(question)


get_rag_answer("Who is Cristiano Ronaldo?")
