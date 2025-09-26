import os
import uuid
import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings
from fpdf import FPDF
import json

# ----------------- Setup -----------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PERSISTENCE_FILE = "processed_files.json"

chroma_client = Client(Settings())
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- Persistence -----------------
def load_processed_files():
    if os.path.exists(PERSISTENCE_FILE):
        with open(PERSISTENCE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_processed_files(mapping):
    with open(PERSISTENCE_FILE, "w") as f:
        json.dump(mapping, f)

# ----------------- File Processing -----------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text, max_chunk_chars=1000, overlap_chars=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_chars, chunk_overlap=overlap_chars)
    return splitter.split_text(text)

def process_file(uploaded_file):
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
        st.error("Unsupported file type.")
        return None

    if not text.strip():
        st.warning(f"No text extracted from {uploaded_file.name}.")
        return None

    chunks = chunk_text(text)
    if not chunks:
        st.warning("Text is empty after chunking.")
        return None

    embeddings = embed_model.encode(chunks).tolist()
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
        st.error(f"Error deleting collection: {e}")

# ----------------- Paper Generation -----------------
def generate_long_research_paper_streaming(topic, min_words=3000):
    sections = ["Abstract", "Introduction", "Related Work", "Methodology",
                "Experiments", "Results", "Discussion", "Conclusion"]
    paper_text = ""
    paper_placeholder = st.empty()

    for sec in sections:
        prompt_text = f"""
        Write a detailed, technical, academic {sec} section on the topic: "{topic}".
        Make it at least {min_words} words long. Cite imaginary references as needed.
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, streaming=True)
        section_text = ""

        def token_callback(token: str):
            nonlocal section_text
            section_text += token
            paper_placeholder.text(paper_text + f"{sec}\n{section_text}")

        model.generate(prompt_text, stream_callback=token_callback)
        paper_text += f"{sec}\n{section_text}\n\n"

    return paper_text

def save_text_as_pdf(text, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="AI Research Paper Analyzer", layout="wide")
    st.title("AI Research Paper Analyzer")

    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = load_processed_files()

    tab_upload, tab_list, tab_generate = st.tabs(["Upload Files", "Uploaded Files", "Generate Paper"])

    # -------- Upload Tab --------
    with tab_upload:
        st.header("Upload Research Papers")
        uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf","doc","docx"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state["processed_files"]:
                    unique_filename = process_file(uploaded_file)
                    if unique_filename:
                        st.success(f"Uploaded and processed: {uploaded_file.name}")
                        st.session_state["processed_files"][uploaded_file.name] = unique_filename
                        save_processed_files(st.session_state["processed_files"])

    # -------- List Tab --------
    with tab_list:
        st.header("Uploaded Files")
        processed_files = st.session_state.get("processed_files", {})
        if processed_files:
            for original_name, unique_filename in list(processed_files.items()):
                st.write(f"**{original_name}**")
                if st.button(f"Delete {original_name}", key=f"delete_{unique_filename}"):
                    delete_file(unique_filename)
                    del st.session_state["processed_files"][original_name]
                    save_processed_files(st.session_state["processed_files"])
                    st.success(f"Deleted {original_name}")
        else:
            st.info("No files uploaded yet.")

    # -------- Generate Paper Tab --------
    with tab_generate:
        st.header("Generate Research Paper")
        topic = st.text_input("Enter research paper topic:")
        min_words = st.slider("Approximate minimum words per section", 300, 2000, 500, step=100)
        if st.button("Generate Paper"):
            if topic.strip():
                paper_text = generate_long_research_paper_streaming(topic, min_words)
                pdf_filename = f"{topic.replace(' ','_')}_research_paper.pdf"
                save_text_as_pdf(paper_text, pdf_filename)
                st.success("Paper generation complete!")
                st.download_button("Download PDF", data=open(pdf_filename,"rb").read(), file_name=pdf_filename)
                st.subheader("Full Generated Paper")
                st.text_area("Paper", value=paper_text, height=400)
            else:
                st.warning("Enter a valid topic.")

if __name__ == "__main__":
    main()
