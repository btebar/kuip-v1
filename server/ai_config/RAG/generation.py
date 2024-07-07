from typing import List, Dict
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from doc_retriever import DocumentRetriever


class RAGGenerator:
    def __init__(self, retriever: DocumentRetriever, model_name: str = "gpt-3.5-turbo"):
        self.retriever = retriever
        self.llm = OpenAI(model_name=model_name)

    def setup_qa_chain(self):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever.vector_store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs
        )

    def generate_answer(self, query: str) -> str:
        if not hasattr(self, 'qa_chain'):
            raise ValueError(
                "QA chain not initialized. Call setup_qa_chain() first.")
        return self.qa_chain.run(query)
