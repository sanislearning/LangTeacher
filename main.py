import os
import traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #Tries to split on bigger boundries, then smaller ones
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st #handles the UI for the chat
import tempfile #used for temporarily storing the files that you upload

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


def load_and_split(file_path):
    loader=PyPDFLoader(file_path) #loads file
    docs=loader.load() #reads and parses the file
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100) #Splits the text into chunks of size 500 charcters
    return splitter.split_documents(docs) #Each chucnk is converted back into a Document object before being returned

def store_chunks(chunks):
    try:
        st.info("🔄 Loading embedding model...")
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        st.success("✅ Embedding model loaded successfully")

        st.info(f"🔄 Embedding {len(chunks)} chunks...")
        vectordb = FAISS.from_documents(chunks, embedding)
        st.success("✅ FAISS index created")
        return vectordb

    except Exception as e:
        st.error("❌ Error in store_chunks:")
        traceback.print_exc()
        raise e



def create_rag_chain(vectordb):
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=1)
    retriever=vectordb.as_retriever(search_type='similarity',k=5)#similarity sets it to use cosine similarity, k=5 tells to pick the top 5 most relevant chunks
    system_prompt=(
        "Use the given context to answer the question."
        "Your role is to be a teacher so students can clear their doubts."
        "Give easy to understand and helpful explanations."
        "Ensure that the answer is clear, concise and easy to understand. Context: {context}"
    )
    prompt=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ('human',"{input}")
    ])

    qa_chain=create_stuff_documents_chain(llm,prompt) #feeds all of the chunks into the prompt together
    return create_retrieval_chain(retriever,qa_chain)
    
#Main function
st.set_page_config(page_title="LangTutor",layout="centered") #sets browser tab title and centers the layout of the entire page
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []  # [{"role": "user"/"assistant", "content": "..."}]
st.title("AI Tutor - Ask Questions from your PDF") #This displays a big heading of course
uploaded_file=st.file_uploader("Upload a PDF file",type=["pdf"]) #Adds a file upload button to the UI
if uploaded_file and "vectordb" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path=tmp_file.name

    try:
        with st.spinner("📖 Reading and splitting document..."):
            chunks=load_and_split(tmp_path)
        vectordb=store_chunks(chunks)
        rag_chain=create_rag_chain(vectordb)

        #Storing in session so that it's not recomputed again and again
        st.session_state.vectordb=vectordb
        st.session_state.rag_chain=rag_chain

        st.success("✅ Document processed. Ask your questions below!")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        traceback.print_exc()

if "rag_chain" in st.session_state:
    question=st.chat_input("Ask a question about the document")

    if question:
        # Add the new user message to memory
        st.session_state.chat_memory.append({"role": "user", "content": question})

        # Create a single string from memory
        history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_memory
        )

        with st.spinner("Thinking..."):
            # Feed full memory into the RAG chain
            response = st.session_state.rag_chain.invoke({"input": history})
            answer = response["answer"]

            # Add assistant response to memory
            st.session_state.chat_memory.append({"role": "assistant", "content": answer})

            # Show in chat UI
            st.chat_message("user").markdown(question)
            st.chat_message("assistant").markdown(answer)


