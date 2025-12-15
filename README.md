# 🧬 AI Recruiter RAG System

A local, AI-powered technical interview system built with Streamlit, ChromaDB, and Ollama. This application ingests PDF job descriptions, generates tailored interview questions using RAG (Retrieval-Augmented Generation), and evaluates candidate responses with scoring and feedback.

## 🌟 Features

*   **📄 PDF Ingestion Pipeline:** Extracts text from Job Description PDFs.
*   **🧠 RAG Architecture:** Uses `chromadb` for vector storage and retrieval.
*   **🤖 Local AI Models:** Powered by Ollama (Llama 3.2 & Nomic Embeddings) for privacy and zero cost.
*   **📊 Interactive Dashboards:** Visualizes chunking strategies and interview performance using Plotly.
*   **🎙️ Automated Interviewer:** Generates context-aware questions and scores answers on Technical Skills, Communication, and Relevance.

## 🛠️ Prerequisites

Before running the Python code, you must have **Ollama** installed and running locally.

1.  **Download Ollama:** [https://ollama.com/download](https://ollama.com/download)
2.  **Pull Required Models:**
    Open your terminal/command prompt and run the following commands to download the specific models used in the code:
    
    ```bash
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    ```

    *Note: These command are done Once !.*     

    After that you only type 
    ```bash
    ollama serve
    ```

## 📦 Installation

1.  **Clone/Download this repository.**
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  **Start Ollama:** Ensure the Ollama app is running in the background (served at `localhost:11434`).
2.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
3.  **In the Browser:**
    *   Upload a Job Description PDF in the "Setup" tab.
    *   Click "Execute Pipeline" to process the file.
    *   Navigate to the "Interview" tab to answer generated questions.
    *   View your detailed analytics in the "Results" tab.

## 📂 Project Structure

*   `app.py`: Main application logic.
*   `chroma_interview_db/`: Automatically created folder storing vector embeddings.