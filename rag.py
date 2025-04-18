import os
import streamlit as st
import sys
import subprocess

# Check and install required packages
def install_packages():
    try:
        import langchain_community
    except ImportError:
        st.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "langchain", "langchain-community", 
                             "faiss-cpu", "huggingface-hub", 
                             "sentence-transformers"])
        st.success("Packages installed successfully! The app will restart now.")
        st.experimental_rerun()

# Run the package installation check at startup
install_packages()

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile

# Set your API token here
DEFAULT_API_TOKEN = ""

# Keep your existing helper functions
def load_document(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        document = loader.load()
        return document
    else:
        raise ValueError("Unsupported file format. Currently only PDF is supported.")

def split_document(document, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(document)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def setup_rag(vector_store, huggingface_api_token):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

def query_document(qa_chain, query):
    result = qa_chain({"query": query})
    return {
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }

# Streamlit app
st.set_page_config(page_title="Document Q&A with RAG", layout="wide")
st.title("Document Q&A with RAG")

# Sidebar for API token
with st.sidebar:
    st.header("Configuration")
    use_custom_token = st.checkbox("Use your own API token", value=False)
    
    if use_custom_token:
        api_token = st.text_input("HuggingFace API Token", type="password")
        st.markdown("**Note:** Your API token is required to use the LLM.")
    else:
        api_token = DEFAULT_API_TOKEN
        st.info("Using application's built-in API token")
    
    # Adding some instructions
    st.header("How to use")
    st.markdown("""
    1. Upload a PDF document
    2. Wait for processing
    3. Ask questions about the document
    """)

# Main area
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:
    if not api_token:
        st.error("Please enter your HuggingFace API Token in the sidebar.")
    else:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with st.spinner("Processing document..."):
            try:
                # Process the document
                document = load_document(tmp_path)
                chunks = split_document(document)
                vector_store = create_vector_store(chunks)
                qa_chain = setup_rag(vector_store, api_token)
                
                st.success("Document processed successfully!")
                st.session_state['qa_chain'] = qa_chain
                
                # Display document info
                st.subheader("Document Information")
                st.write(f"Filename: {uploaded_file.name}")
                st.write(f"Document chunks: {len(chunks)}")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up the temporary file
                os.unlink(tmp_path)

    # Query section
    if 'qa_chain' in st.session_state:
        st.subheader("Ask Questions")
        query = st.text_input("Type your question:")
        
        if st.button("Submit Question"):
            if query:
                with st.spinner("Generating answer..."):
                    try:
                        result = query_document(st.session_state['qa_chain'], query)
                        
                        # Display answer
                        st.subheader("Answer")
                        st.write(result["answer"])
                        
                        # Display sources
                        with st.expander("View Sources"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content[:300] + "...")
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question.")

# Instructions when no file is uploaded
else:
    st.info("Please upload a PDF document to get started.")

# Add some styling and footer
st.markdown("---")
st.markdown("RAG-powered document Q&A system using LangChain and HuggingFace models.")