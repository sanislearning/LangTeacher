import os
import traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #Tries to split on bigger boundries, then smaller ones
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


def load_and_split(file_path):
    loader=PyPDFLoader(file_path) #loads file
    docs=loader.load() #reads and parses the file
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100) #Splits the text into chunks of size 500 charcters
    print("1")
    return splitter.split_documents(docs) #Each chucnk is converted back into a Document object before being returned

def store_chunks(chunks):
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("hi once more")
        vectordb = Chroma.from_documents(chunks, embedding, persist_directory="db")
        print("hi")
        vectordb.persist()
        print("2")
        return vectordb
    except Exception as e:
        print("❌ Error in store_chunks:")
        traceback.print_exc()  # Shows complete error
        raise e


def create_rag_chain(vectordb):
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.1)
    retriever=vectordb.as_retriever(search_type='similarity',k=5)#similarity sets it to use cosine similarity, k=5 tells to pick the top 5 most relevant chunks
    system_prompt=(
        "Use the given context to answer the question."
        "If you don't know the answer, say you don't know."
        "Ensure that the answer is clear, concise and easy to understand. Context: {context}"
    )
    prompt=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ('human',"{input}")
    ])

    qa_chain=create_stuff_documents_chain(llm,prompt) #feeds all of the chunks into the prompt together
    print('3')
    return create_retrieval_chain(retriever,qa_chain)
    
#Main function
if __name__=="__main__":
    try:
        file_path="thebook.pdf"
        chunks=load_and_split(file_path)
        vectordb=store_chunks(chunks)
        rag=create_rag_chain(vectordb)
        while(True):
            question=input("Ask any questions")
            response=rag.invoke({"input":question})
            print("Tutor says: ",response['answer'])
            cont=int(input("Click 1 if you'd like to continue, else click any other value"))
            if (cont!=1):
                break
    except Exception as e:
        print(e)