import os
import uuid
import streamlit as st
from pathlib import Path
from fpdf import FPDF

# --- Document Extraction Libraries ---
import fitz  # PyMuPDF
from docx import Document
try:
    import pdfplumber
    USE_PDFPLUMBER = True
except ImportError:
    USE_PDFPLUMBER = False

# --- LangChain & Google ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# ---------- Google API Key (directly in code) ----------
GOOGLE_API_KEY = "AIzaSyA5tNFi6kjgpRMP43spbAur8lsSwfAHSug"

# ---------- Upload Directory ----------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------- Document Extraction -----------------
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
                    full_text += page.extract_text() + "\n"
        except Exception:
            full_text = extract_text_from_pdf_pymupdf(file_path)
    else:
        full_text = extract_text_from_pdf_pymupdf(file_path)
    return full_text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# ----------------- Text Chunking -----------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# ----------------- Vector Store -----------------
def create_vector_store(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="textembedding-gecko-001", api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="textembedding-gecko-001", api_key=GOOGLE_API_KEY)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ----------------- Research Paper Generator -----------------
def generate_research_paper(topic, min_words=1000):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, api_key=GOOGLE_API_KEY)
    prompt_template = """
    Write a detailed research paper on "{topic}".
    Include Abstract, Introduction, Related Work, Methodology, Results, Discussion, and Conclusion.
    The paper should be at least {min_words} words.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "min_words"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain.run({"topic": topic, "min_words": min_words})

def save_text_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="Research Paper Analyzer", layout="wide")
    st.title("AI Research Paper Analyzer")

    tab_gen, tab_chat = st.tabs(["Research Paper Generator", "Upload & Chat with PDF"])

    # ----------------- Research Paper Generator Tab -----------------
    with tab_gen:
        st.header("Generate Research Paper")
        topic = st.text_input("Enter your research paper topic:")
        min_words = st.slider("Minimum words", 1000, 8000, 3000, step=500)
        if st.button("Generate Paper"):
            if topic.strip() == "":
                st.warning("Please enter a topic")
            else:
                with st.spinner("Generating research paper..."):
                    paper_text = generate_research_paper(topic, min_words)
                    pdf_filename = f"{topic.replace(' ', '_')}.pdf"
                    save_text_as_pdf(paper_text, pdf_filename)
                    st.success("Research paper generated!")
                    st.download_button("Download PDF", data=open(pdf_filename, "rb").read(), file_name=pdf_filename)

    # ----------------- Upload & Chat Tab -----------------
    with tab_chat:
        st.header("Upload PDFs & Chat")
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf", "docx"], accept_multiple_files=True)
        if uploaded_files:
            all_text = ""
            for uploaded_file in uploaded_files:
                ext = uploaded_file.name.split(".")[-1].lower()
                file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{uploaded_file.name}")
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if ext == "pdf":
                    all_text += extract_text_from_pdf(file_path)
                elif ext in ["docx", "doc"]:
                    all_text += extract_text_from_docx(file_path)

            chunks = get_text_chunks(all_text)
            vector_store = create_vector_store(chunks)
            st.success("PDF processed! You can now ask questions.")

        question = st.text_input("Ask a question from the PDF")
        if question and Path("faiss_index").exists():
            vector_store = load_vector_store()
            docs = vector_store.similarity_search(question)
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, api_key=GOOGLE_API_KEY)
            prompt_template = """
            Answer the question based on the following context:
            {context}
            Question: {question}
            Answer:
            """
            context = "\n".join([doc.page_content for doc in docs])
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = LLMChain(llm=model, prompt=prompt)
            answer = chain.run({"context": context, "question": question})
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()
