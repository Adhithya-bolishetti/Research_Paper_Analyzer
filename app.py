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

# --- Google Generative AI ---
import google.generativeai as genai

# ---------- Global Setup ----------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PERSISTENCE_FILE = "processed_files.json"
chroma_client = Client(Settings())
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Google API Key ----------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("Google API Key not found in secrets.toml")

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
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

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

    chunks = chunk_text_improved(text)
    if not chunks:
        st.warning("The extracted text is empty after chunking.")
        return None

    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embed_model.encode(batch_chunks).tolist()
        embeddings.extend(batch_embeddings)

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

# ---------- Google LLM Call for Research Paper ----------
def generate_research_paper(topic, min_words=3000):
    if not GOOGLE_API_KEY:
        return "Google API Key missing"
    prompt = f"""
    Write a detailed academic research paper on the topic: "{topic}".
    Include Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, and Conclusion.
    The paper should be at least {min_words} words.
    Use formal language and cite imaginary references if needed.
    """
    try:
        response = genai.chat.create(
            model="gemini-1.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.last
    except Exception as e:
        return f"Error generating paper: {e}"

def save_text_as_pdf(text, filename):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

# ---------- Streamlit App ----------
def main():
    st.title("AI Research Paper & PDF Analyzer")

    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = load_processed_files()

    tab_generate, tab_chat = st.tabs(["Research Paper Generator", "Upload & Chat with PDF"])

    # --- Research Paper Generator Tab ---
    with tab_generate:
        st.header("Generate a Research Paper")
        topic = st.text_input("Enter your research paper topic:")
        min_words = st.slider("Minimum words", 1000, 8000, 3000, step=500)
        if st.button("Generate Paper"):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating research paper..."):
                    paper_text = generate_research_paper(topic, min_words)
                    pdf_filename = "generated_research_paper.pdf"
                    save_text_as_pdf(paper_text, pdf_filename)
                    st.success("Paper generated!")
                    st.download_button("Download PDF", data=open(pdf_filename, "rb").read(), file_name=pdf_filename)

    # --- Upload & Chat with PDF Tab ---
    with tab_chat:
        st.header("Upload PDFs/DOCs for Q&A")
        uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf","docx"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state["processed_files"]:
                    unique_filename = process_file(uploaded_file)
                    if unique_filename:
                        st.success(f"Uploaded: {uploaded_file.name}")
                        st.session_state["processed_files"][uploaded_file.name] = unique_filename
                        save_processed_files(st.session_state["processed_files"])

        st.subheader("Ask a Question")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if query:
                search_results = search_documents(query)
                context = "\n\n".join([doc for _, doc, _ in search_results[:5]]) if search_results else ""
                if context:
                    # Use Google API for answering too
                    answer_prompt = f"Answer based only on the context below. If not available, say 'No relevant information'.\n\nContext:\n{context}\n\nQuestion: {query}"
                    try:
                        response = genai.chat.create(
                            model="gemini-1.5-turbo",
                            messages=[{"role": "user", "content": answer_prompt}],
                            temperature=0.3
                        )
                        answer = response.last
                    except Exception as e:
                        answer = f"Error: {e}"
                    st.write("**Answer:**")
                    st.write(answer)
                else:
                    st.warning("No relevant documents found.")

if __name__ == "__main__":
    main()
