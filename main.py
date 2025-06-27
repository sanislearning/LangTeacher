import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #Tries to split on bigger boundries, then smaller ones
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import retrieval_qa #LangChain wrapper for RAG, combines a retriever(Chroma here) with a chat model

load_dotenv()
os.environ["GOOGLE-API-KEY"]=os.getenv("GOOGLE-API-KEY")


def load_and_split(file_path):
    loader=UnstructuredFileLoader(file_path) #loads unstructured file
    docs=loader.load() #reads and parses the file
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100) #Splits the text into chunks of size 500 charcters
    return splitter.split_documents(docs) #Each chucnk is converted back into a Document object before being returned

def store_chunks(chunks):
    embedding=GoogleGenerativeAIEmbeddings(model='models/embeddings-001') #Declares the embedding model here
    vectordb=Chroma.from_documents(chunks,embedding,persist_directory="db")
    vectordb.persist()
    return vectordb

def create_rag_chain(vectordb):
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.1)
    retriever=vectordb.as_retriever(search_type='similarity',k=5)#similarity sets it to use cosine similarity, k=5 tells to pick the top 5 most relevant chunks
    return retrieval_qa.from_chain_type(llm=llm,retriever=retriever) #Creates a RAG chain using LangChain's retrieval_qa utility
    #from_chain_type sets up the whole pipeline with the llm and the retriever

#Main function
if __name__=="__main__":
    file_path="thebook.pdf"
    chunks=load_and_split(file_path)
    vectordb=store_chunks(chunks)
    rag=create_rag_chain(vectordb)

    question="Explain what ML is."
    print("Tutor says:",rag.run(question))
