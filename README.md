# AI-Powered PDF Chatbot using Gemini and LangChain

This project is a smart PDF chatbot built using **Google Gemini LLM**, **LangChain**, **FAISS**, and **Streamlit**. It enables users to upload a PDF and ask natural language questions about its content. The app parses the uploaded PDF, splits it into manageable chunks, embeds those chunks using Google Generative AI embeddings, and performs semantic search to answer user queries with the help of Gemini's powerful language model.

---

## Features

- Upload and process any PDF file.
- Extracts and chunks text intelligently using LangChain.
- Uses Google’s `embedding-001` model for vector embedding.
- Employs FAISS for fast and relevant semantic search.
- Integrates Gemini 2.5 Flash for accurate and intelligent answers.
- Interactive web interface powered by Streamlit.

---

## Tech Stack

- **Streamlit** – for building the user interface.
- **PyPDF2** – to extract text from uploaded PDF documents.
- **LangChain** – for text splitting and chaining large language models.
- **FAISS** – for efficient similarity-based document retrieval.
- **Google Gemini LLM** – to answer questions based on document context.
- **GoogleGenerativeAIEmbeddings** – to embed the text into vectors for semantic search.

---

## Project Structure

- `app.py` – Main application file containing all logic.
- `requirements.txt` – List of dependencies.
- `.gitignore` – To exclude unnecessary files from Git.
- `README.md` – Project overview and setup instructions.
