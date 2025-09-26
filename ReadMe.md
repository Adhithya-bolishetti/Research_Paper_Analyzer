# Research Paper Analyzer

This project provides tools for generating, uploading, and analyzing research papers using AI-powered language models and vector search. It features two main applications:

- **app.py**: A Streamlit app for generating research papers with Gemini, uploading PDFs, and chatting with their content.
- **rag.py**: A Streamlit app for uploading, chunking, embedding, and querying research papers using ChromaDB and OpenRouter LLMs.

## Features

- Generate structured research papers on any topic.
- Upload PDF/DOCX files and extract their content.
- Chunk and embed documents for semantic search.
- Ask questions and get answers based on uploaded documents.
- Summarize uploaded PDFs.

## Project Structure

```
.
├── secrets.toml
├── app.py
├── rag.py
└── requirements.txt

```

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up API keys**:
    - Add your Google API key to `secrets.toml`:
      ```
      GOOGLE_API_KEY=your_google_api_key
      ```
    - Add your OpenRouter API key and site info to `secrets.toml`.

4. **Run the apps**:
    - For the main analyzer:
      ```sh
      streamlit run app.py
      ```
    - For the RAG app:
      ```sh
      streamlit run rag.py
      ```

## Usage

- **Generate Research Paper**: Use the "Generate Research Paper" tab to create a new paper on any topic.
- **Upload & Chat with PDF**: Upload PDF files and ask questions about their content.
- **RAG App**: Use the RAG app for advanced document upload, search, and Q&A.

## Requirements

See [requirements.txt](requirements.txt) for the full list of dependencies.

## License

This project is for research and educational purposes.

---

**Note:** Ensure you have valid API keys for Google Generative AI and OpenRouter to use all features.
