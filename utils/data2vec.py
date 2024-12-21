'''python libraries'''
from typing import List
import pandas as pd
import os

'''Langchain'''
from langchain_chroma import Chroma
from chromadb import Client
from chromadb.config import Settings

from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

'''faiss'''
import faiss


def etl_data(file_path):
    '''
    Loads csv file to python's dataframe and preprocessing data
    1. handles file not found
    2. handles missing null values -> drops strategy

    Args:
        file_path: file location in the system
        returns : python's dataframe
    '''
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError
        df = pd.read_csv(file_path, header='infer')
        #Remove null values
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(e)
        return None
    
def etl_txt(file_path):
    '''
    Loads text files and returns as string

    Args:
        file_path: file location in the system
        returns string
    '''
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError
        txt = ''
        with open(file_path, 'r', encoding='latin-1') as f:
            file = f.readline()
            txt += file
        return txt
    except Exception as e:
        print(e)
        return None
    
class CSVLoaderWithDropna(BaseLoader):
    """Loads a CSV file and drops rows with null values.

    Args:
        file_path (str): Path to the CSV file.
        page_content_column (str): Name of the column to use for page content.
        metadata_columns (list[str], optional): List of columns to use as metadata.
                                               Defaults to all columns except
                                               `page_content_column`.
    """

    def __init__(self, file_path: str, page_content_column: str,
                 metadata_columns: list = None):
        self.file_path = file_path
        self.page_content_column = page_content_column
        self.metadata_columns = metadata_columns

    def load(self) -> list[Document]:
        """Load the CSV file and create a list of Documents."""
        try:
            df = etl_data(self.file_path) # Drop rows with null values

            if self.metadata_columns is None:
                # Use all columns except page_content_column as metadata
                self.metadata_columns = [col for col in df.columns
                                        if col != self.page_content_column]
            
            documents = []
            for index in range(len(df)):
                row = df.iloc[index, :]
                page_content = row[self.page_content_column]
                metadata = {col: row[col] for col in self.metadata_columns}
                metadata['source'] = csv_file_path
                metadata['row'] = index
                documents.append(Document(page_content=page_content, metadata=metadata))
            return documents
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []

def txt_file_loader(file_path, metadata) -> List[Document]:
    """Loads data from a text file.

    Args:
        file_path: Path to the text file.
        metadata: Metadata to associate with the document.

    Returns:
        A list of Documents.
    """
    try:
        loader = TextLoader(file_path, encoding='latin-1') # Specify encoding if needed
        documents = loader.load()
        
        if metadata:
          for doc in documents:
              doc.metadata.update(metadata)
        return documents
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

def create_document(file_path):
    loader = CSVLoaderWithDropna(
        file_path=file_path,
        page_content_column='program modules',
        metadata_columns=['Course code','descri','course name','outcomes_x'])
    return loader.load()

#Load Data
dir = "data"
csv_file_path = "merged_humber_courses.csv"
text_file_path = "Humber_Guide_Developing_Learning_outcomes.txt"

#course-link,course name,Program code,sem,outcomes_x,req,program modules,descri,Course code
course_data = etl_data(os.path.join(dir, csv_file_path))
guide_data = etl_txt(os.path.join(dir, text_file_path))

#Create DOCUMENTS
course_docs = create_document(os.path.join(dir, csv_file_path))
guide_docs = txt_file_loader(os.path.join(dir, text_file_path), metadata={'source':text_file_path})

#Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#Emedding funciton
embedding_func = OpenAIEmbeddings(model='text-embedding-3-small')

#list of chunks
course_splits = text_splitter.split_documents(course_docs)
guide_splits = text_splitter.split_documents(guide_docs)

embedding_size = len(embedding_func.embed_query(
    ' '.join(course_data['descri'].tolist()) + guide_data
))

#Make index for Memory
index = faiss.IndexFlatIP(embedding_size)
docstore = InMemoryDocstore({})
index_to_docstore_id = {}

#vectore store
try:
    vectorstore = FAISS(
        index = index,
        embedding_function=embedding_func,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    vectorstore.add_documents(course_splits)
    vectorstore.add_documents(guide_splits)
    vectorstore.save_local('vector_db')
    
except Exception as e:
    print(e)