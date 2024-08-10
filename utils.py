from langchain.chains import RetrievalQA
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from config import API_KEY


class LoadDocument:
    def __init__(self) -> None:
        pass

    def load_pdf(self,path):
        """
        load pdf document and return it
        """
        loader = PyPDFLoader(path)
        return loader.load()
    


class ProcessText:
    def __init__(self) -> None:
        pass


    def document_split(self,document,chunk_size:int, chunk_overlap:int):
        """
        for spliting the langchain document object and split into chunks.
        return chunks
        """
        spliter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
        return spliter.split_documents(documents=document)
    
    def text_split(self,document,chunk_size:int, chunk_overlap:int):
        """
        for spliting the langchain text object intoand split into chunks.
        return chunks
        """
        spliter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
        return spliter.split_text(documents=document)
    


class LLM:
    def __init__(self) -> None:
        pass

    def embedding(self,model_name:str = "models/embedding-001"):
        """Load google embedding model."""
        return GoogleGenerativeAIEmbeddings(model= model_name, google_api_key=API_KEY)

    def llm_model(self,model_name:str = "gemini-pro",temperature:int = 0):
        """Load google gemini model"""
        return GoogleGenerativeAI(model=model_name,google_api_key=API_KEY,temperature=temperature)


class DocRetrievel:
    def __init__(self) -> None:
        pass

    def keyword_retriever(self,document, k:int=3):
        """
        Initialize Keyword retriever pipeline
        """

        keyword_retrieveral = BM25Retriever.from_documents(documents= document)
        keyword_retrieveral.k = k
        return keyword_retrieveral
    
    def hybrid_retriever(self,document):
        """
        Combine Keyword retriever and vector retriever
        """
        keyword_retrieveral = self.keyword_retriever(document=document)
        vector_retrieveral = VectorStore().ChromaDB(document=document)
        hybrid_retriever = EnsembleRetriever(retrievers=[vector_retrieveral,keyword_retrieveral],weights=[0.3,0.7])


class VectorStore:
    def __init__(self) -> None:
        pass

    def ChromaDB(self,document,path:str, k:int = 3):
        """
        Store vectors into chromadb database and return vector retriever
        """
        vector_store = Chroma.from_documents(embedding=self.embeddings, documents= document)        
        return vector_store.as_retriever(search_kwargs  = {"k":k})


class Chain:
    def __init__(self) -> None:
        self.llm = LLM().llm_model()
        self.chain_type = "stuff"

    def retriever_qa_chain(self,retriever,query:str):
        """
        create object for Retrieval QA Chain.
        """
        hybrid_chain =  RetrievalQA.from_chain_type(llm = self.llm, chain_type=self.chain_type, retriever = retriever)
        return hybrid_chain.invoke(query)