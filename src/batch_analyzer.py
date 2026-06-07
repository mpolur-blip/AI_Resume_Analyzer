import os
import pdfplumber
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESUME_DIR = os.path.join(BASE_DIR, "sample_data", "resumes")
JD_DIR = os.path.join(BASE_DIR, "sample_data", "job_descriptions")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


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


def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def find_skills(text):
    found_skills = []
    lower_text = text.lower()

    for skill in SKILL_KEYWORDS.keys():
        if skill.lower() in lower_text:
            found_skills.append(skill)

    return found_skills


def calculate_keyword_score(resume_text, jd_text):
    documents = [resume_text, jd_text]

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(documents)

    similarity_score = cosine_similarity(matrix)[0][1]

    return round(similarity_score * 100, 2)


def calculate_skill_score(matched_skills, jd_skills):
    if len(jd_skills) == 0:
        return 0

    return round((len(matched_skills) / len(jd_skills)) * 100, 2)


def calculate_education_score(resume_text, jd_text):
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    education_keywords = [
        "bachelor",
        "master",
        "computer science",
        "engineering",
        "degree"
    ]

    jd_requirements = []

    for keyword in education_keywords:
        if keyword in jd_lower:
            jd_requirements.append(keyword)

    if len(jd_requirements) == 0:
        return 100

    matched_requirements = []

    for keyword in jd_requirements:
        if keyword in resume_lower:
            matched_requirements.append(keyword)

    return round((len(matched_requirements) / len(jd_requirements)) * 100, 2)


def calculate_overall_score(keyword_score, skill_score, education_score):
    overall_score = (
        0.40 * keyword_score +
        0.40 * skill_score +
        0.20 * education_score
    )

    return round(overall_score, 2)


def clean_filename(filename):
    return filename.replace(".pdf", "").replace(".txt", "")


def run_batch_analysis():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    resume_files = [
        file for file in os.listdir(RESUME_DIR)
        if file.endswith(".pdf")
    ]

    jd_files = [
        file for file in os.listdir(JD_DIR)
        if file.endswith(".txt")
    ]

    all_results = []

    for resume_file in resume_files:
        resume_path = os.path.join(RESUME_DIR, resume_file)
        resume_text = extract_text_from_pdf(resume_path)
        resume_skills = find_skills(resume_text)

        for jd_file in jd_files:
            jd_path = os.path.join(JD_DIR, jd_file)
            jd_text = read_text_file(jd_path)
            jd_skills = find_skills(jd_text)

            matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
            missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

            keyword_score = calculate_keyword_score(resume_text, jd_text)
            skill_score = calculate_skill_score(matched_skills, jd_skills)
            education_score = calculate_education_score(resume_text, jd_text)

            overall_score = calculate_overall_score(
                keyword_score,
                skill_score,
                education_score
            )

            all_results.append({
                "Resume_Name": clean_filename(resume_file),
                "Job_Description": clean_filename(jd_file),
                "Overall_ATS_Score": overall_score,
                "Keyword_Score": keyword_score,
                "Skill_Score": skill_score,
                "Education_Score": education_score,
                "Matched_Skills": ", ".join(matched_skills),
                "Missing_Skills": ", ".join(missing_skills),
                "Matched_Skill_Count": len(matched_skills),
                "Missing_Skill_Count": len(missing_skills),
                "Total_JD_Skills": len(jd_skills)
            })

    results_df = pd.DataFrame(all_results)

    output_path = os.path.join(RESULTS_DIR, "batch_resume_jd_scores.csv")
    results_df.to_csv(output_path, index=False)

    print("Batch analysis completed!")
    print(f"Total comparisons: {len(results_df)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    run_batch_analysis()
    