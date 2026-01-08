import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory."""
    try:
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def process_file(file_path, api_key, endpoint, embedding_deployment, api_version):
    """Load text, split it, and create vector store."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key
    )
    
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def get_answer_chain(vector_store, api_key, endpoint, chat_deployment, api_version):
    """Create a retrieval chain."""
    llm = AzureChatOpenAI(
        azure_deployment=chat_deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=0
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain

def main():
    st.set_page_config(page_title="Azure AI Document Q&A", layout="wide")
    st.title("Azure AI Document Q&A Assistant")

    # Initialize session state for vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for Configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("API Key", type="password")
        endpoint = st.text_input("Endpoint")
        deployment_name = st.text_input("Deployment Name (Chat)")
        embedding_deployment = st.text_input("Deployment Name (Embeddings)")
        api_version = st.text_input("API Version", value="2023-05-15")
        
        st.divider()
        st.subheader("Document Upload")
        uploaded_file = st.file_uploader("Upload a file (PDF/TXT)", type=["pdf", "txt"])
        
        if uploaded_file and st.button("Process Document"):
            if not (api_key and endpoint and embedding_deployment):
                 st.error("Please fill in all Azure OpenAI settings.")
            else:
                with st.spinner("Processing..."):
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        st.session_state.vector_store = process_file(
                            file_path, api_key, endpoint, embedding_deployment, api_version
                        )
                        st.success("Document Processed Successfully!")

    # Main Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.vector_store is None:
                response = "Please upload and process a document first."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif not (api_key and endpoint and deployment_name):
                 response = "Please configure Azure OpenAI settings."
                 st.markdown(response)
                 st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.spinner("Thinking..."):
                    try:
                        qa_chain = get_answer_chain(
                            st.session_state.vector_store, 
                            api_key, endpoint, deployment_name, api_version
                        )
                        result = qa_chain.invoke({"query": prompt})
                        response = result['result']
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

