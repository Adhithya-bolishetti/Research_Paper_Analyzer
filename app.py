import os
import uuid
import streamlit as st
from fpdf import FPDF
from docx import Document
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI  # Or replace with any local LLM if available
from langchain.prompts import PromptTemplate

# ---------- Global Setup ----------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Local embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Utility Functions ----------
def save_text_as_pdf(text: str, filename: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = [embed_model.encode(chunk).tolist() for chunk in chunks]
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def generate_research_paper(topic: str, min_words=1000):
    """
    Simple text generator using OpenAI LLM (replace with your preferred LLM)
    """
    llm = OpenAI(temperature=0.3)  # replace with your local or API-based LLM
    prompt_template = """
    Write a detailed academic research paper on the topic: "{topic}".
    Include Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion.
    Paper length: at least {min_words} words. Use formal academic style.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "min_words"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain.run({"input_documents": [], "question": f"Generate research paper on {topic} of {min_words} words"})

# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="Research Paper & PDF QA", layout="wide")
    st.title("AI Research Paper Generator & PDF Q&A")

    tab_paper, tab_pdf = st.tabs(["Research Paper Generator", "Upload & Chat with PDF"])

    # ----- Research Paper Generator Tab -----
    with tab_paper:
        st.header("Generate a Research Paper")
        topic = st.text_input("Enter research paper topic:")
        min_words = st.slider("Minimum words", 500, 5000, 1500, step=100)
        if st.button("Generate Paper"):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating research paper..."):
                    paper_text = generate_research_paper(topic, min_words)
                    pdf_filename = f"{topic.replace(' ','_')}.pdf"
                    save_text_as_pdf(paper_text, pdf_filename)
                    st.success("Research paper generated!")
                    st.download_button("Download PDF", data=open(pdf_filename, "rb").read(), file_name=pdf_filename)

    # ----- Upload & Chat with PDF Tab -----
    with tab_pdf:
        st.header("Upload PDF/DOCX and Ask Questions")
        uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf","docx"], accept_multiple_files=True)
        if uploaded_files:
            all_chunks = []
            for file in uploaded_files:
                ext = file.name.split('.')[-1].lower()
                temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.name}")
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                if ext == "pdf":
                    text = extract_text_from_pdf(temp_path)
                else:
                    text = extract_text_from_docx(temp_path)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
            if all_chunks:
                vector_store = create_vector_store(all_chunks)
                st.success("Files processed and vector store created!")

                question = st.text_input("Ask a question about the uploaded files:")
                if question:
                    docs = vector_store.similarity_search(question, k=5)
                    llm = OpenAI(temperature=0)  # Replace with preferred LLM
                    prompt = PromptTemplate(
                        template="Answer the question based on the context below.\nContext:\n{context}\nQuestion: {question}\nAnswer:",
                        input_variables=["context","question"]
                    )
                    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
                    answer = chain.run({"input_documents": docs, "question": question})
                    st.write("**Answer:**")
                    st.write(answer)

if __name__ == "__main__":
    main()
