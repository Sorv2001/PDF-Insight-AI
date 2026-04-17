import streamlit as st
import os
import re
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from tempfile import NamedTemporaryFile

# 1. Configuration and Setup
load_dotenv()
st.set_page_config(page_title="Sourav RAG Book Assistant", page_icon="📄")

def clean_text(text):
    return re.sub(r'[\ud800-\udfff]', '', text)

# 2. Sidebar: File Upload
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    process_btn = st.button("Process Document")

# 3. Processing Logic
if uploaded_file and process_btn:
    with st.spinner("Processing PDF... This may take a moment."):
        # Save uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load and Clean
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Embed and Store in Memory (or persistent path)
        embeddings = MistralAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory="./chroma_db_streamlit"
        )
        
        # Save to session state
        st.session_state.vectorstore = vectorstore
        st.success("Document processed! You can now ask questions.")
        os.remove(tmp_path) # Clean up temp file

# 4. Chat Interface
st.title("📄 PDF RAG Assistant")
st.caption("Chat with your documents using Mistral AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt_input := st.chat_input("Ask something about the document..."):
    # Check if vectorstore exists
    if "vectorstore" not in st.session_state:
        st.error("Please upload and process a PDF first!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # RAG Logic
        with st.chat_message("assistant"):
            # Setup Retriever
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
            )
            
            # Get Context
            relevant_docs = retriever.invoke(prompt_input)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            # LLM Setup
            llm = ChatMistralAI(model="mistral-small-2402")
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Use ONLY the provided context to answer. If not found, say you don't know."),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])

            # Generate Response
            chain = qa_prompt | llm
            response = chain.invoke({"context": context, "question": prompt_input})
            
            full_response = response.content
            st.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})