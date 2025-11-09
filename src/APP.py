import os 
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling - ONLY CHANGED USER MESSAGE TO RADISH ORANGE
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    /* Light gray background for the main content - KEEPING AS BEFORE */
    .main .block-container {
        background-color: #f5f5f5;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .chat-container {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    /* CHANGED: User question to radish orange background */
    .user-message {
        background: linear-gradient(135deg, #FF5349 0%, #FF355E 100%);
        color: #000000;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.8rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(255, 83, 73, 0.3);
        font-size: 1rem;
        line-height: 1.5;
        font-weight: 500;
    }
    /* AI response to white background with black text */
    .bot-message {
        background: #FFFFFF;
        color: #000000;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.8rem 0;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        font-size: 1rem;
        line-height: 1.6;
        border: 1px solid #e0e0e0;
        font-weight: 400;
    }
    .typing-indicator {
        display: inline-block;
        padding: 12px 20px;
        background: #EAf7FF;
        border-radius: 18px;
        font-style: italic;
        color: #5D6D7E;
        border-left: 4px solid #4A90E2;
    }
    .status-success {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        text-align: center;
        font-weight: 600;
        margin: 1rem auto;
        max-width: 300px;
        box-shadow: 0 2px 10px rgba(39, 174, 96, 0.3);
    }
    .stChatInput {
        border-radius: 20px;
        border: 2px solid #4A90E2;
        padding: 12px;
    }
    .connect-btn {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border: none;
        padding: 16px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: block;
        margin: 2rem auto;
        box-shadow: 0 4px 15px 0 rgba(46, 139, 87, 0.3);
    }
    .connect-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(46, 139, 87, 0.4);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid #2E8B57;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
    }
    .clear-btn {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">⚖️ Legal AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Intelligent Legal Research Partner • Powered by FAISS & Groq Llama 3.1</div>', unsafe_allow_html=True)

# Load environment variables
load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables")
    st.stop()

# Initialize LLM
LLM = ChatGroq(model='llama-3.1-8b-instant', groq_api_key=groq_api_key)

# FIXED: Improved prompt template without "Based on context" prefix
prompt = ChatPromptTemplate.from_template(
    """You are a knowledgeable legal assistant specializing in Indian laws. Answer the question using ONLY the provided context. 
Provide a natural, comprehensive answer that directly addresses the question without starting with phrases like "Based on the context" or "According to the context".

Context: {context}

Question: {input}

Answer in a clear, professional manner as if you're a legal expert:"""
)

# RAG initialization function with FAISS caching
def RAG_function():
    if 'vectors' not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en')
        faiss_index_path = "faiss_index"

        # If saved FAISS index exists, load it safely
        if os.path.exists(faiss_index_path):
            st.session_state.vectors = FAISS.load_local(
                faiss_index_path,
                st.session_state.embedding,
                allow_dangerous_deserialization=True
            )
        else:
            # Load and split the PDF
            st.session_state.loader = PyPDFLoader('legal_document.pdf')
            st.session_state.doc = st.session_state.loader.load()
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=300
            )
            st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.doc)
            
            # Create FAISS vector store and save it
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc, st.session_state.embedding)
            st.session_state.vectors.save_local(faiss_index_path)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_connected" not in st.session_state:
    st.session_state.db_connected = False

# Database Connection Section
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if not st.session_state.db_connected:
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <h3 style='color: #2C3E50; margin-bottom: 1rem;'>🚀 Ready to Explore Legal Knowledge</h3>
            <p style='color: #5D6D7E; font-size: 1.1rem;'>Connect to our comprehensive legal database to get started with your research</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔗 Connect to Legal Database", key="connect_db", use_container_width=True):
            with st.spinner("🔄 Initializing legal database and loading documents..."):
                RAG_function()
                st.session_state.db_connected = True
                st.markdown('<div class="status-success">✅ Database Connected Successfully!</div>', unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()

# Chat Interface (only show if database is connected)
if st.session_state.db_connected:
    st.markdown("---")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">⚖️ {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt_text := st.chat_input("Ask your legal question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        
        # Display user message immediately
        with chat_container:
            st.markdown(f'<div class="user-message">👤 {prompt_text}</div>', unsafe_allow_html=True)
            
            # Display typing indicator
            with st.empty():
                st.markdown('<div class="typing-indicator">⚖️ Researching legal documents...</div>', unsafe_allow_html=True)
                
                # Get RAG response
                try:
                    # Create retriever
                    retriever = st.session_state.vectors.as_retriever(search_kwargs={'k': 6})
                    
                    # Create document chain with improved prompt
                    doc_chain = create_stuff_documents_chain(LLM, prompt)
                    
                    # Create retrieval chain
                    retriever_chain = create_retrieval_chain(retriever, doc_chain)
                    
                    # Invoke the chain
                    response = retriever_chain.invoke({'input': prompt_text})
                    answer = response['answer']
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"I apologize, but I'm having trouble accessing the legal database right now. Please try again in a moment."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Rerun to update the chat display
        st.rerun()

    # Sidebar for document sources
    if st.session_state.messages:
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-content">
                <h3>📚 Legal Resources</h3>
                <p>Access retrieved legal document sections and research materials</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources for the last response if available
            if 'response' in locals() and 'context' in response:
                with st.expander("🔍 View Source Documents", expanded=False):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Document Section {i+1}**")
                        st.info(doc.page_content[:350] + "..." if len(doc.page_content) > 350 else doc.page_content)
                        st.markdown("---")

            # Clear chat button
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

else:
    # Show features when not connected
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <h3 style='color: #2C3E50; margin-bottom: 2rem;'>🌟 Why Choose Legal AI Assistant?</h3>
        <div style='display: flex; justify-content: center; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;'>
            <div class="feature-card" style='flex: 1; min-width: 250px; max-width: 280px;'>
                <h4 style='color: #2E8B57;'>🔍 Intelligent Search</h4>
                <p style='color: #5D6D7E;'>Advanced semantic search through comprehensive legal documents with precise context understanding</p>
            </div>
            <div class="feature-card" style='flex: 1; min-width: 250px; max-width: 280px;'>
                <h4 style='color: #2E8B57;'>⚡ Instant Responses</h4>
                <p style='color: #5D6D7E;'>Get immediate answers powered by Groq's high-speed AI inference technology</p>
            </div>
            <div class="feature-card" style='flex: 1; min-width: 250px; max-width: 280px;'>
                <h4 style='color: #2E8B57;'>📚 Legal Expertise</h4>
                <p style='color: #5D6D7E;'>Specialized knowledge in Indian constitutional law and legal frameworks</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7F8C8D; margin-top: 3rem; padding: 1rem; font-size: 0.9rem;'>"
    "Built with ❤️ using Streamlit, LangChain, and Groq • Legal AI Assistant v2.0 • Your Trusted Legal Research Partner"
    "</div>",
    unsafe_allow_html=True
)