import os
import uuid
import json
import requests
import streamlit as st
from pathlib import Path

# --- Document Extraction Libraries ---
import fitz  # PyMuPDF for PDF extraction
from docx import Document  # DOCX extraction
try:
    import pdfplumber
    USE_PDFPLUMBER = True
except ImportError:
    USE_PDFPLUMBER = False

# --- Embedding Model ---
from sentence_transformers import SentenceTransformer

# --- ChromaDB ---
from chromadb import Client
from chromadb.config import Settings

# ---------- Global Setup ----------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PERSISTENCE_FILE = "processed_files.json"
chroma_client = Client(Settings())
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Persistence Functions ----------
def load_processed_files():
    if os.path.exists(PERSISTENCE_FILE):
        with open(PERSISTENCE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_processed_files(mapping):
    with open(PERSISTENCE_FILE, "w") as f:
        json.dump(mapping, f)

# ---------- Document Extraction ----------
def extract_text_from_pdf_pymupdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_text_from_pdf(file_path):
    full_text = ""
    if USE_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    table_text = ""
                    tables = page.extract_tables() or []
                    for table in tables:
                        for row in table:
                            table_text += "\t".join([str(cell) for cell in row if cell]) + "\n"
                    full_text += page_text + "\n" + table_text + "\n"
        except Exception:
            full_text = extract_text_from_pdf_pymupdf(file_path)
    else:
        full_text = extract_text_from_pdf_pymupdf(file_path)
    return full_text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# ---------- Chunking ----------
def chunk_text_improved(text, max_chunk_chars=1000, overlap_chars=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                current_chunk = overlap + para + "\n\n"
            else:
                current_chunk = para[:max_chunk_chars]
                chunks.append(current_chunk.strip())
                current_chunk = para[max_chunk_chars:]
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

# ---------- Processing Uploaded Files ----------
def process_file(uploaded_file, batch_size=64):
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    if file_extension.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_extension.lower() in [".doc", ".docx"]:
        text = extract_text_from_docx(file_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

    if not text.strip():
        st.warning(f"No text could be extracted from {uploaded_file.name}.")
        return None

    # Chunk text
    chunks = chunk_text_improved(text)
    if not chunks:
        st.warning("The extracted text is empty after chunking.")
        return None

    # Batch embeddings
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embed_model.encode(batch_chunks).tolist()
        embeddings.extend(batch_embeddings)

    # Handle existing collection
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if unique_filename in existing_collections:
        collection = chroma_client.get_collection(name=unique_filename)
    else:
        collection = chroma_client.create_collection(name=unique_filename)

    doc_ids = [str(i) for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=doc_ids)
    
    return unique_filename

def delete_file(unique_filename):
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    try:
        chroma_client.delete_collection(name=unique_filename)
    except Exception as e:
        st.warning(f"Error deleting collection: {e}")

# ---------- Search ----------
def search_documents(query, top_k=5):
    results = []
    try:
        collections = chroma_client.list_collections()
    except Exception as e:
        st.error(f"Failed to list collections: {e}")
        return results

    if not collections:
        st.info("No documents uploaded yet.")
        return results

    for col in collections:
        try:
            coll = chroma_client.get_collection(name=col.name)
            search_result = coll.query(query_texts=[query], n_results=top_k)
            docs = search_result.get("documents", [[]])[0]
            distances = search_result.get("distances", [[]])[0]
            for doc, dist in zip(docs, distances):
                results.append((col.name, doc, dist))
        except Exception as e:
            st.warning(f"Error querying collection '{col.name}': {e}")

    results.sort(key=lambda x: x[2])
    return results

# ---------- LLM Call ----------
def call_llm(context, question):
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("OpenRouter API key not provided in secrets.")
        return "API Key Missing"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": st.secrets.get("SITE_URL", "https://example.com"),
        "X-Title": st.secrets.get("SITE_NAME", "My Site"),
        "Content-Type": "application/json"
    }
    message = (
        "You are an AI assistant that answers questions solely based on the provided context from uploaded documents. "
        "If the context does not contain relevant information, respond: 'No relevant information found in the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    data = {"model": "qwen/qwq-32b:free", "messages": [{"role": "user", "content": message}]}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Error from API: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Request failed: {e}"

# ---------- Streamlit App ----------
def main():
    st.title("AI Research Paper Summarizer")
    st.write("Upload PDFs/DOCs, ask questions, and get answers from your documents.")

    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = load_processed_files()

    tab_upload, tab_list, tab_prompt, tab_chat = st.tabs(["File Upload", "Uploaded Files", "Prompt", "Upload & Chat"])

    # --- File Upload Tab ---
    with tab_upload:
        st.header("Upload Research Papers")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files", type=["pdf", "doc", "docx"], accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state["processed_files"]:
                    unique_filename = process_file(uploaded_file)
                    if unique_filename:
                        st.success(f"Uploaded and processed: {uploaded_file.name}")
                        st.session_state["processed_files"][uploaded_file.name] = unique_filename
                        save_processed_files(st.session_state["processed_files"])

    # --- Uploaded Files Tab ---
    with tab_list:
        st.header("Uploaded Files")
        processed_files = st.session_state.get("processed_files", {})
        if processed_files:
            for original_name, unique_filename in list(processed_files.items()):
                st.write(f"**{original_name}**")
                if st.button(f"Delete {original_name}", key=f"delete_{unique_filename}"):
                    delete_file(unique_filename)
                    st.success(f"Deleted {original_name}")
                    del st.session_state["processed_files"][original_name]
                    save_processed_files(st.session_state["processed_files"])
        else:
            st.info("No files uploaded yet.")

    # --- Prompt Tab ---
    with tab_prompt:
        st.header("Ask a Question")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if query:
                search_results = search_documents(query)
                if search_results:
                    context = "\n\n".join([doc for _, doc, _ in search_results[:5]])
                else:
                    context = ""
                if not context:
                    st.error("No relevant content found from uploaded documents.")
                else:
                    answer = call_llm(context, query)
                    st.write("**Answer:**")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")

    # --- Upload & Chat Tab ---
    with tab_chat:
        st.header("Upload & Chat with PDF")
        chat_uploaded_file = st.file_uploader(
            "Upload a PDF for Q&A (not stored)", type=["pdf"]
        )
        if chat_uploaded_file:
            temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4().hex}.pdf")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(chat_uploaded_file.getbuffer())

            extracted_text = extract_text_from_pdf(temp_file_path)
            os.remove(temp_file_path)

            if not extracted_text.strip():
                st.warning("No text could be extracted.")
            else:
                user_question = st.text_input("Enter your question:")
                if user_question:
                    chunks = chunk_text_improved(extracted_text)
                    embeddings = embed_model.encode(chunks).tolist()
                    temp_collection_name = f"temp_{uuid.uuid4().hex}"
                    collection = chroma_client.create_collection(name=temp_collection_name)
                    doc_ids = [str(i) for i in range(len(chunks))]
                    collection.add(documents=chunks, embeddings=embeddings, ids=doc_ids)

                    search_result = collection.query(query_texts=[user_question], n_results=5)
                    relevant_chunks = search_result.get("documents", [[]])[0]

                    if not relevant_chunks:
                        st.warning("No relevant information found in PDF.")
                    else:
                        context = "\n\n".join(relevant_chunks)
                        answer = call_llm(context, user_question)
                        st.write("**Answer:**")
                        st.write(answer)

                    try:
                        chroma_client.delete_collection(name=temp_collection_name)
                    except Exception as e:
                        st.warning(f"Failed to delete temporary collection: {e}")

if __name__ == "__main__":
    main
