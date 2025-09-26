import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
api_key = st.secrets.get("GOOGLE_API_KEY")
# genai.configure(api_key=google_api_key)

def generate_long_research_paper(topic: str, min_words=3000) -> str:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
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

def save_text_as_pdf(text: str, filename: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="textembedding-gecko-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say, "answer is not available in the context". Do not provide wrong answers.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.chat_history.append((user_question, response["output_text"]))
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Research Paper Analyzer", layout="wide")
    st.title("Research paper Analyzer")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar menu
    menu = st.sidebar.selectbox("Choose an action", ["Generate Research Paper", "Upload & Chat with PDF"])

    if menu == "Generate Research Paper":
        st.header("Generate a Research Paper")
        topic = st.text_input("Enter your research paper topic:")
        min_words = st.slider("Minimum words (approximate)", 1000, 8000, 3000, step=500)

        if st.button("Generate Paper"):
            if not topic.strip():
                st.warning("Please enter a valid topic.")
            else:
                with st.spinner("Generating research paper... This may take a minute or two."):
                    paper_text = generate_long_research_paper(topic, min_words)
                    pdf_filename = "generated_research_paper.pdf"
                    save_text_as_pdf(paper_text, pdf_filename)
                    st.success(f"Research paper generated and saved as {pdf_filename}.")
                    st.download_button("Download PDF", data=open(pdf_filename, "rb").read(), file_name=pdf_filename)

    elif menu == "Upload & Chat with PDF":
        st.header("Upload PDF Files and Chat")

        pdf_docs = st.file_uploader("Upload your PDF files here (multiple allowed):", accept_multiple_files=True)
        if st.button("Process PDF(s)"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDF(s)..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")

        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input(user_question)

        st.subheader("Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {question}")
            st.markdown(f"**A{i+1}:** {answer}")
            st.markdown("---")

if __name__ == "__main__":
    main()
