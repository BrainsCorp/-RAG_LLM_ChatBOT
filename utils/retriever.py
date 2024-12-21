
# from chromadb import Client
# from chromadb.config import Settings
# from langchain.vectorstores import Chroma
# from langchain_core.runnables import chain
from typing import List
import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# import faiss


class LangChainRetriever:
    def __init__(self, vector_db_path):
        #OpenAI embeddings
        self.embedding_func = OpenAIEmbeddings(model='text-embedding-3-small')
        self.vector_db_filepath = vector_db_path
        self.vector_db = self.load_vector_db(self.vector_db_filepath)
    
    def load_vector_db(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError

            vector_db = FAISS.load_local(
                folder_path=file_path,
                embeddings=self.embedding_func,
                allow_dangerous_deserialization=True
            )

            return vector_db

        except Exception as e:
            print(f"Error loading Faiss index: {e}")

    def query_retriever(self, query: str, filter:dict=None, threshold=0.3) -> List[Document]:
        search_kwargs = {'k': 10,
                         'search_type':'similarity'}
         
        results = self.vector_db.similarity_search_with_score(
            query=query,
            search_kwargs=search_kwargs,
            filter=filter
        )

        filtered_docs = []
        for doc, score in results:
            if score > threshold:
                filtered_docs.append(doc)

        if not results:
            print("No Docs related to query!")
            return []
        
        
        return results

# Usage
# retriever = LangChainRetriever(vector_db_path='vector_db')
# response = retriever.query_retriever("Bloom Taxanomy", filter={'source':'Humber_Guide_Developing_Learning_outcomes.txt'})
# for doc, score in response:
#     print(doc.page_content)