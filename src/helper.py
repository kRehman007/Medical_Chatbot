from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


#Function to load data from Pdf file
def load_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    documents=loader.load()
    return documents



#Fiter function to extract only metadata and page_content
def custom_filter(complete_doc:List[Document])->List[Document]:
    filtered_doc:List[Document]=[]
    for doc in complete_doc:
        src=doc.metadata.get("source")
        filtered_doc.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return filtered_doc


#Splitting filtered document into smaller chunks for avoiding context window error(token limit)
def split_doc(doc):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Maximum characters per chunk
    chunk_overlap=20  # Overlap between chunks
    )
    chunks = text_splitter.split_documents(doc)
    return chunks


#Function for downlaoding embedding object from hugging face...
def download_embedding():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    )
    return hf
    
