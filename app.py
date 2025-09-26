import streamlit as st
from fpdf import FPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# ---------- Research Paper Generation Function ----------
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

# ---------- Streamlit Tab ----------
def research_paper_generator_tab():
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
