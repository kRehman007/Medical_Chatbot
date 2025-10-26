from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from prompt import custom_prompt
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")


from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    api_key=groq_api_key,
    timeout=None,
    max_retries=2,
    )

prompt = ChatPromptTemplate.from_template(custom_prompt)


def get_medical_chain(query,retriever):
    combine_docs_chain = create_stuff_documents_chain(
    llm,prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    response = retrieval_chain.invoke({"input": query})
    resp=response["answer"]
    return resp