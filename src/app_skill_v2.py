import os
import pdfplumber
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("AI Resume Analyzer")


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_JD_DIR = os.path.join(BASE_DIR, "sample_data", "job_descriptions")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# -------------------------------------------------
# Load sample job descriptions
# -------------------------------------------------
def get_sample_jd_files():
    if not os.path.exists(SAMPLE_JD_DIR):
        return []

    jd_files = []

    for file in os.listdir(SAMPLE_JD_DIR):
        if file.endswith(".txt"):
            jd_files.append(file)

    return sorted(jd_files)


def load_jd_file(filename):
    filepath = os.path.join(SAMPLE_JD_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


# -------------------------------------------------
# Skill dictionary
# -------------------------------------------------
SKILL_KEYWORDS = {
    "Python": "Programming",
    "Java": "Programming",
    "C++": "Programming",
    "JavaScript": "Programming",
    "TypeScript": "Programming",

    "React": "Frontend",
    "Next JS": "Frontend",
    "HTML": "Frontend",
    "CSS": "Frontend",

    "SQL": "Database",
    "Pandas": "Data Analysis",
    "NumPy": "Data Analysis",

    "Machine Learning": "AI/ML",
    "Deep Learning": "AI/ML",
    "TensorFlow": "AI/ML",
    "Keras": "AI/ML",
    "Scikit-learn": "AI/ML",
    "LSTM": "AI/ML",
    "CNN": "AI/ML",

    "Power BI": "Visualization",
    "Tableau": "Visualization",
    "Excel": "Visualization",

    "Software Testing": "QA",
    "Test Cases": "QA",
    "Bug Tracking": "QA",
    "Quality Assurance": "QA",
    "QA Engineer": "QA",

    "Docker": "DevOps",
    "Dockerfile": "DevOps",
    "Git": "Tools",
    "GitHub": "Tools",

    "JSON": "Data Formats",
    "CSV": "Data Formats",
    "API": "Backend",
    "REST API": "Backend",

    "Statistics": "Statistics",
    "Outliers": "Statistics",
    "Correlation": "Statistics",
    "Anomaly Detection": "AI/ML",

    "Benchmark": "AI Evaluation",
    "SWE-bench": "AI Evaluation",
    "Terminal-Bench": "AI Evaluation",

    "Multi-agent": "Agentic AI",
    "Agents": "Agentic AI",

    "Communication": "Soft Skills",
    "Problem Solving": "Soft Skills",
    "Analytical Skills": "Soft Skills",
    "Leadership": "Soft Skills",
    "Time Management": "Soft Skills",
    "Team Collaboration": "Soft Skills"
}


# -------------------------------------------------
# PDF text extraction
# -------------------------------------------------
def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text


# -------------------------------------------------
# Skill extraction
# -------------------------------------------------
def find_skills(text):
    found_skills = []
    lower_text = text.lower()

    for skill in SKILL_KEYWORDS.keys():
        if skill.lower() in lower_text:
            found_skills.append(skill)

    return found_skills


# -------------------------------------------------
# Priority assignment
# -------------------------------------------------
def get_priority(skill):
    high_priority = [
        "Python",
        "SQL",
        "Pandas",
        "NumPy",
        "Docker",
        "Statistics",
        "Anomaly Detection",
        "Machine Learning",
        "Scikit-learn"
    ]

    medium_priority = [
        "Git",
        "GitHub",
        "JSON",
        "CSV",
        "Benchmark",
        "Multi-agent",
        "Agents",
        "TensorFlow",
        "Power BI",
        "Tableau"
    ]

    if skill in high_priority:
        return "High"
    elif skill in medium_priority:
        return "Medium"
    else:
        return "Low"


# -------------------------------------------------
# Smart recommendation generator
# -------------------------------------------------
def get_recommendation(skill):
    recommendations = {
        "Python": "Add Python-based projects involving data processing, automation, backend development, or machine learning.",
        "SQL": "Mention SQL queries, joins, database design, or analysis performed on structured datasets.",
        "Pandas": "Add a data cleaning or exploratory data analysis project using Pandas.",
        "NumPy": "Mention numerical analysis, array operations, or preprocessing tasks performed using NumPy.",
        "Scikit-learn": "Add a machine learning project using Scikit-learn models such as Logistic Regression, Random Forest, or SVM.",
        "TensorFlow": "Mention deep learning models built using TensorFlow, such as LSTM, CNN, or neural networks.",
        "Keras": "Add details about neural network models implemented using Keras.",
        "Machine Learning": "Add a project where you trained and evaluated machine learning models using real datasets.",
        "Deep Learning": "Mention deep learning architectures such as CNN, LSTM, or Transformer models.",
        "Statistics": "Highlight statistical analysis, distributions, correlations, hypothesis testing, or outlier detection.",
        "Power BI": "Mention dashboards, KPI reports, or visual analytics created using Power BI.",
        "Tableau": "Add dashboarding or data visualization experience using Tableau.",
        "Excel": "Mention Excel-based analysis, pivot tables, formulas, or reporting experience.",
        "Docker": "Add a project where you containerized an application using Docker.",
        "Dockerfile": "Mention writing Dockerfiles to build and run reproducible application environments.",
        "Git": "Mention version control experience using Git commands, branching, and commits.",
        "GitHub": "Add GitHub repositories with clean README files, project structure, and documented results.",
        "API": "Add experience building or consuming APIs using Flask, FastAPI, or REST services.",
        "REST API": "Mention building or integrating REST APIs in backend or full-stack projects.",
        "JSON": "Mention working with JSON files, API responses, or structured data exchange.",
        "CSV": "Mention cleaning and analyzing CSV datasets using Python or SQL.",
        "React": "Add a frontend project built using React components and reusable UI design.",
        "Next JS": "Mention building web applications using Next.js and routing.",
        "JavaScript": "Add frontend or backend scripting projects using JavaScript.",
        "TypeScript": "Mention strongly typed frontend development using TypeScript.",
        "HTML": "Mention web page structure and UI development using HTML.",
        "CSS": "Mention responsive design and styling using CSS.",
        "Software Testing": "Add experience with manual testing, regression testing, or test planning.",
        "Test Cases": "Mention writing and executing test cases for web or mobile applications.",
        "Bug Tracking": "Mention reporting and tracking defects using tools like Jira, Trello, or GitHub Issues.",
        "Quality Assurance": "Add QA-related project experience focused on software quality and reliability.",
        "QA Engineer": "Highlight testing workflows, defect reporting, and collaboration with developers.",
        "Communication": "Highlight written communication, documentation, presentations, or stakeholder collaboration.",
        "Problem Solving": "Mention examples where you solved technical issues, debugged code, or improved workflows.",
        "Analytical Skills": "Add examples of analyzing data, identifying patterns, and making evidence-based decisions.",
        "Leadership": "Mention leadership roles, team coordination, event organization, or mentoring experience.",
        "Time Management": "Mention managing deadlines, multitasking, or completing projects within time constraints.",
        "Team Collaboration": "Highlight teamwork in academic projects, internships, or group-based development work.",
        "Anomaly Detection": "Add a project involving detection of unusual patterns in logs, transactions, or sensor data.",
        "Benchmark": "Mention evaluation, comparison, or benchmarking of models, systems, or algorithms.",
        "SWE-bench": "Mention familiarity with AI coding benchmarks or software engineering evaluation tasks.",
        "Terminal-Bench": "Mention experience with terminal-based coding environments or AI benchmark tasks.",
        "Multi-agent": "Add exposure to multi-agent workflows, AI agents, or task decomposition.",
        "Agents": "Mention AI agents, tool-using systems, or automated reasoning workflows."
    }

    return recommendations.get(
        skill,
        f"Add or strengthen experience related to {skill} in your resume."
    )


# -------------------------------------------------
# Score calculations
# -------------------------------------------------
def calculate_skill_score(matched_skills, jd_skills):
    if len(jd_skills) == 0:
        return 0

    return round((len(matched_skills) / len(jd_skills)) * 100, 2)


def calculate_education_score(resume_text, job_description):
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    education_keywords = [
        "bachelor",
        "master",
        "computer science",
        "engineering",
        "degree"
    ]

    jd_education_requirements = []

    for keyword in education_keywords:
        if keyword in jd_lower:
            jd_education_requirements.append(keyword)

    if len(jd_education_requirements) == 0:
        return 100

    matched_requirements = []

    for keyword in jd_education_requirements:
        if keyword in resume_lower:
            matched_requirements.append(keyword)

    return round((len(matched_requirements) / len(jd_education_requirements)) * 100, 2)


def calculate_overall_ats_score(keyword_score, skill_score, education_score):
    overall_score = (
        0.40 * keyword_score +
        0.40 * skill_score +
        0.20 * education_score
    )

    return round(overall_score, 2)


# -------------------------------------------------
# Sidebar sample JD selector
# -------------------------------------------------
sample_jd_files = get_sample_jd_files()

st.sidebar.header("Sample Job Descriptions")

selected_jd = st.sidebar.selectbox(
    "Choose a sample JD",
    ["Paste manually"] + sample_jd_files
)

default_jd_text = ""

if selected_jd != "Paste manually":
    default_jd_text = load_jd_file(selected_jd)


# -------------------------------------------------
# User inputs
# -------------------------------------------------
uploaded_resume = st.file_uploader("Upload Resume PDF", type=["pdf"])

job_description = st.text_area(
    "Paste Job Description",
    value=default_jd_text,
    height=250
)


# -------------------------------------------------
# Main analysis
# -------------------------------------------------
if uploaded_resume and job_description:

    resume_text = extract_text_from_pdf(uploaded_resume)

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(documents)

    similarity_score = cosine_similarity(matrix)[0][1]
    match_percentage = round(similarity_score * 100, 2)

    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(job_description)

    matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
    missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

    keyword_score = match_percentage
    skill_score = calculate_skill_score(matched_skills, jd_skills)
    education_score = calculate_education_score(resume_text, job_description)

    overall_ats_score = calculate_overall_ats_score(
        keyword_score,
        skill_score,
        education_score
    )

    # ---------------------------------------------
    # ATS score display
    # ---------------------------------------------
    st.subheader("ATS Score Breakdown")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall ATS Score", f"{overall_ats_score}%")

    with col2:
        st.metric("Keyword Match", f"{keyword_score}%")

    with col3:
        st.metric("Skill Match", f"{skill_score}%")

    with col4:
        st.metric("Education Match", f"{education_score}%")

    st.progress(overall_ats_score / 100)

    # ---------------------------------------------
    # Matched and missing skills
    # ---------------------------------------------
    st.subheader("Matched Skills")
    if matched_skills:
        st.write(", ".join(matched_skills))
    else:
        st.write("No major matched skills found.")

    st.subheader("Missing Skills")
    if missing_skills:
        st.write(", ".join(missing_skills))
    else:
        st.write("No major missing skills found.")

    # ---------------------------------------------
    # Resume improvement suggestions
    # ---------------------------------------------
    st.subheader("Resume Improvement Suggestions")

    if missing_skills:
        for skill in missing_skills:
            st.markdown(f"**{skill}:** {get_recommendation(skill)}")
    else:
        st.write("Your resume already covers the major skills detected from this job description.")

    # ---------------------------------------------
    # Table creation
    # ---------------------------------------------
    analysis_rows = []

    for skill in matched_skills:
        analysis_rows.append({
            "Skill": skill,
            "Category": SKILL_KEYWORDS[skill],
            "Status": "Matched",
            "Priority": "Already Present",
            "Recommendation": "Keep this skill clearly visible in your resume.",
            "Keyword_Score": keyword_score,
            "Skill_Score": skill_score,
            "Education_Score": education_score,
            "Overall_ATS_Score": overall_ats_score
        })

    for skill in missing_skills:
        analysis_rows.append({
            "Skill": skill,
            "Category": SKILL_KEYWORDS[skill],
            "Status": "Missing",
            "Priority": get_priority(skill),
            "Recommendation": get_recommendation(skill),
            "Keyword_Score": keyword_score,
            "Skill_Score": skill_score,
            "Education_Score": education_score,
            "Overall_ATS_Score": overall_ats_score
        })

    analysis_df = pd.DataFrame(analysis_rows)

    st.subheader("Skill Analysis Table")
    st.dataframe(analysis_df)

    # ---------------------------------------------
    # Export CSV
    # ---------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "resume_analysis.csv")
    analysis_df.to_csv(csv_path, index=False)

    st.success("Analysis exported to results/resume_analysis.csv")

    # ---------------------------------------------
    # Resume preview
    # ---------------------------------------------
    st.subheader("Extracted Resume Text Preview")
    st.write(resume_text[:2000])

else:
    st.info("Upload a resume PDF and paste or select a job description to begin analysis.")