import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Analyzer")

uploaded_resume = st.file_uploader("Upload Resume PDF", type=["pdf"])

job_description = st.text_area("Paste Job Description")


def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text


if uploaded_resume and job_description:

    resume_text = extract_text_from_pdf(uploaded_resume)

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(documents)
    similarity_score = cosine_similarity(matrix)[0][1]

    st.subheader("Match Score")
    st.write(f"{round(similarity_score * 100, 2)} %")

    st.subheader("Resume Text")
    st.write(resume_text[:2000])