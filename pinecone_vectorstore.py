from dotenv import load_dotenv
import os
from pinecone import Pinecone,ServerlessSpec
from Medical_Chatbot.src.helper import load_pdf_file,custom_filter,split_doc,download_embedding


load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
pinecone_api_key=os.getenv("PINECONE_API_KEY")

print("GROQ_API_KEY",groq_api_key)
print("PINECONE_API_KEY",pinecone_api_key)



pc = Pinecone(api_key=pinecone_api_key)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

from langchain_pinecone import PineconeVectorStore
embeddings=download_embedding()
extracted_text=load_pdf_file("Medical_book.pdf")
filtered_text=custom_filter(extracted_text)
chunked_text=split_doc(filtered_text)
vector_store = PineconeVectorStore.from_documents(
    documents=chunked_text,
    index_name=index_name, 
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3,"score_threshold": 0.5},
)