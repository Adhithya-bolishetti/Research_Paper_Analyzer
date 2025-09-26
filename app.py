import os
import uuid
import json
import streamlit as st
from fpdf import FPDF
from pathlib import Path

# --- Document Extraction Libraries ---
import fitz  # PyMuPDF
from docx import Document
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

# --- LLM for paper generation ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

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
                    full_text += page_text + "\n"
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

# ---------- Research Paper Generator ----------
def generate_research_paper(topic: str, min_words: int) -> str:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt_template = """
    Write a detailed, structured research paper on the topic: "{topic}".
    Include Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, and Conclusion sections.
    Each section should be academic and at least {min_words} words.
    Use formal language and cite imaginary references as needed.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "min_words"])
    chain = LLMChain(llm=model, prompt=prompt)
    paper_text = chain.run({"topic": topic, "min_words": min_words})
    return paper_text

def save_text_as_pdf(text: str, filename: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

def research_paper_tab():
    st.header("Research Paper Generator")
    topic = st.text_input("Enter the research paper topic:")
    min_words = st.slider("Minimum words per section", 300, 2000, 500, step=50)

    if st.button("Generate Paper"):
        if not topic.strip():
            st.warning("Please enter a valid topic.")
            return
        with st.spinner("Generating research paper, please wait..."):
            paper_text = generate_research_paper(topic, min_words)
            pdf_filename = f"{topic.replace(' ','_')}_research_paper.pdf"
            save_text_as_pdf(paper_text, pdf_filename)
        st.success("Research paper generated successfully!")
        st.download_button("Download PDF", data=open(pdf_filename, "rb").read(), file_name=pdf_filename)
        st.subheader("Paper Preview")
        st.text_area("Paper Content", value=paper_text, height=400)

# ---------- Streamlit Main ----------
def main():
    st.title("AI Research Paper Tool")
    tab_generate, tab_chat = st.tabs(["Research Paper Generator", "Upload & Chat with PDF"])

    with tab_generate:
        research_paper_tab()

    with tab_chat:
        st.header("Upload & Chat with PDF")
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = load_processed_files()

        uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf","doc","docx"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state["processed_files"]:
                    unique_filename = process_file(uploaded_file)
                    if unique_filename:
                        st.success(f"Uploaded and processed: {uploaded_file.name}")
                        st.session_state["processed_files"][uploaded_file.name] = unique_filename
                        save_processed_files(st.session_state["processed_files"])

        st.subheader("Ask Questions from Uploaded Files")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if query:
                results = []
                for col in chroma_client.list_collections():
                    collection = chroma_client.get_collection(name=col.name)
                    search_result = collection.query(query_texts=[query], n_results=5)
                    docs = search_result.get("documents", [[]])[0]
                    results.extend(docs)
                context = "\n\n".join(results[:5])
                if not context:
                    st.warning("No relevant information found.")
                else:
                    answer = call_llm(context, query)
                    st.write("**Answer:**")
                    st.write(answer)

if __name__ == "__main__":
    main()
