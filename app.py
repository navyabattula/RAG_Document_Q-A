import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Question: {input}
""")

# Initialize session state
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

def create_vectors_embedding():
    try:
        st.session_state.embeddings = OllamaEmbeddings()
        loader = PyPDFDirectoryLoader("Research")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        return True
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return False

# Streamlit UI
st.title("Research Paper Q&A")

# Document embedding button
if st.button("Create Document Embedding"):
    with st.spinner("Creating vector database..."):
        if create_vectors_embedding():
            st.success("Vector Database is ready!")

# Query input
user_prompt = st.text_input("Enter your query from the research paper")

# Process query
if user_prompt and st.session_state.vectors is not None:
    try:
        # Create the document chain
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )
        
        # Get the retriever
        retriever = st.session_state.vectors.as_retriever()
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
        
        with st.spinner("Processing your query..."):
            start = time.process_time()
            # Execute the chain using invoke instead of calling directly
            response = retrieval_chain.invoke({"input": user_prompt})
            process_time = time.process_time() - start
            
            st.write("### Answer:")
            st.write(response['answer'])
            st.write(f"Processing time: {process_time:.2f} seconds")
            
            with st.expander("Document similarity search"):
                for i, doc in enumerate(response['context']):
                    st.write(f"Document {i+1}:")
                    st.write(doc.page_content)
                    st.write('---')
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
elif user_prompt:
    st.warning("Please create the document embedding first!")
