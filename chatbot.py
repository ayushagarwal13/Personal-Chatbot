import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# Set Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA36dzygiS2nEtVi3rrGzoI25PNmG0deCk"

# Streamlit UI
st.header("My first Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Process PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user question
    user_question = st.text_input("Type Your question here")

    if user_question:
        match = vector_store.similarity_search(user_question)

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        # QA Chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
