import os
import uuid
import json
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

# --- Google LLM ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# --- Embeddings ---
from sentence_transformers import SentenceTransformer

# --- ChromaDB ---
from chromadb import Client
from chromadb.config import Settings

# --- PDF Generation ---
from fpdf import FPDF

# ---------- Global Setup ----------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

chroma_client = Client(Settings())
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Helper Functions ----------
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
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    full_text += (page.extract_text() or "") + "\n"
        except Exception:
            full_text = extract_text_from_pdf_pymupdf(file_path)
    else:
        full_text = extract_text_from_pdf_pymupdf(file_path)
    return full_text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

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

def save_text_as_pdf(text, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

# ---------- Research Paper Generation ----------
def generate_long_research_paper(topic, min_words=3000, api_key=None):
    if not api_key:
        st.error("Google API Key missing! Check secrets.toml")
        return ""
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, api_key=api_key)
    prompt_template = """
    Write a detailed, structured research paper on the topic: "{topic}".
    Include Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, and Conclusion sections.
    The paper should be technical, academic, and at least {min_words} words long.
    Use formal language and cite imaginary references as needed.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "min_words"])
    chain = LLMChain(llm=model, prompt=prompt)
    paper_text = chain.run({"topic": topic, "min_words": min_words})
    return paper_text

# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="AI Research Paper & PDF Analyzer", layout="wide")
    st.title("AI Research Paper & PDF Analyzer")

    api_key = st.secrets.get("GOOGLE_API_KEY")
    
    tab_gen, tab_chat = st.tabs(["Research Paper Generator", "Upload & Chat with PDF"])

    # --- Research Paper Generator ---
    with tab_gen:
        st.header("Generate Research Paper")
        topic = st.text_input("Enter your research paper topic:")
        min_words = st.slider("Minimum words (approx.)", 1000, 8000, 3000, step=500)

        if st.button("Generate Paper"):
            if not topic.strip():
                st.warning("Please enter a valid topic.")
            else:
                with st.spinner("Generating research paper..."):
                    paper_text = generate_long_research_paper(topic, min_words, api_key)
                    if paper_text:
                        pdf_filename = f"Research_Paper_{uuid.uuid4().hex[:6]}.pdf"
                        save_text_as_pdf(paper_text, pdf_filename)
                        st.success("Research paper generated!")
                        st.download_button("Download PDF", data=open(pdf_filename, "rb").read(), file_name=pdf_filename)

    # --- Upload & Chat with PDF ---
    with tab_chat:
        st.header("Upload PDFs/DOCs and Ask Questions")
        uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf","docx"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded. You can ask questions now.")
        
        user_question = st.text_input("Ask a question from your uploaded documents:")
        if user_question and uploaded_files:
            all_text = ""
            for file in uploaded_files:
                ext = os.path.splitext(file.name)[1].lower()
                if ext == ".pdf":
                    all_text += extract_text_from_pdf(file) + "\n"
                elif ext in [".doc", ".docx"]:
                    all_text += extract_text_from_docx(file) + "\n"

            chunks = chunk_text_improved(all_text)
            embeddings = GoogleGenerativeAIEmbeddings(model="textembedding-gecko-001", api_key=api_key)
            # Create temporary FAISS index
            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            docs = vector_store.similarity_search(user_question)
            # Use Google LLM to answer
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, api_key=api_key)
            prompt_text = "\n\n".join([doc.page_content for doc in docs])
            chain_prompt = f"Answer the question based on the context below:\n\nContext:\n{prompt_text}\n\nQuestion:\n{user_question}\nAnswer:"
            answer = model(prompt_text + "\n\nQuestion: " + user_question)
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()