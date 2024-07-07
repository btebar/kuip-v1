from doc_retriever import DocumentRetriever
from generation import RAGGenerator


# Set up the retriever
retriever = DocumentRetriever("./test_data")
retriever.setup()

# Set up the generator
generator = RAGGenerator(retriever)
generator.setup_qa_chain()


def get_rag_answer(question: str) -> str:
    return generator.generate_answer(question)


get_rag_answer("Who is Cristiano Ronaldo?")
