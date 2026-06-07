# AI Resume Analyzer

An AI-powered resume analysis web application that compares a resume PDF with a job description, calculates an ATS-style match score, identifies matched and missing skills, and provides resume improvement suggestions.

## Project Overview

This project helps users evaluate how well a resume matches a specific job description. The application extracts text from a resume PDF, compares it with a job description using NLP techniques, detects important skills, and generates an ATS score breakdown.

The goal of this project is to demonstrate practical NLP, resume-job matching, skill gap analysis, and interactive web app development using Python and Streamlit.

## Features

* Upload resume in PDF format
* Select sample job descriptions from a dropdown
* Paste custom job descriptions manually
* Extract resume text using `pdfplumber`
* Calculate keyword similarity using TF-IDF and cosine similarity
* Detect matched skills between resume and job description
* Detect missing skills required by the job description
* Generate ATS-style score breakdown:

  * Overall ATS Score
  * Keyword Match Score
  * Skill Match Score
  * Education Match Score
* Provide resume improvement suggestions for missing skills
* Export analysis results to CSV for future dashboarding
* Includes sample resumes and sample job descriptions for testing

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* pdfplumber
* TF-IDF Vectorization
* Cosine Similarity

## Project Structure

```text
AI_Resume_Analyzer/
│
├── sample_data/
│   ├── resumes/
│   │   ├── resume_data_scientist.pdf
│   │   ├── resume_ml_engineer.pdf
│   │   ├── resume_qa_engineer.pdf
│   │   └── ...
│   │
│   └── job_descriptions/
│       ├── jd_data_scientist.txt
│       ├── jd_ml_engineer.txt
│       ├── jd_qa_engineer.txt
│       └── ...
│
├── src/
│   ├── app.py
│   └── app_skill_v2.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## How It Works

1. The user uploads a resume PDF.
2. The user selects or pastes a job description.
3. The app extracts resume text using `pdfplumber`.
4. TF-IDF vectorization converts the resume and job description into numerical vectors.
5. Cosine similarity calculates the keyword match score.
6. A predefined skill dictionary is used to detect matched and missing skills.
7. The app calculates an ATS-style score using keyword, skill, and education scores.
8. Missing skills are converted into resume improvement suggestions.
9. Results are displayed in the Streamlit app and exported as a CSV file.

## ATS Score Formula

The overall ATS score is calculated using:

```text
Overall ATS Score =
40% Keyword Match Score
+ 40% Skill Match Score
+ 20% Education Match Score
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mpolur-blip/AI_Resume_Analyzer.git
cd AI_Resume_Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run src/app_skill_v2.py
```

## Sample Data

This repository includes dummy sample resumes and job descriptions for testing. These files are not real resumes and do not contain private personal information.

Example test cases:

* Data Science resume + Data Science JD → higher match score
* QA resume + Data Science JD → lower match score
* Backend resume + AI Engineer JD → moderate or low match score depending on overlapping skills
## Power BI Dashboard

This project also includes a Power BI dashboard that visualizes resume-job matching results across multiple sample resumes and job descriptions.

The dashboard was created using batch comparison data generated from:

```bash
python src/batch_analyzer.py
```

The batch analyzer compares all sample resumes against all sample job descriptions and produces 100 resume-job comparisons.

### Dashboard Pages

#### 1. Overview

The Overview page includes:

* Total resume-job comparisons
* Average ATS score
* Best ATS score
* Average skill match
* Resume filter
* Job description filter
* Resume vs Job Match Matrix

#### 2. Resume Performance

The Resume Performance page includes:

* Average ATS score by resume
* Average ATS score by job description
* Average missing skills by job description
* Keyword match vs skill match scatter plot

### Dashboard File

The Power BI dashboard is available in:

```text
dashboard/AI_Resume_Analyzer_Dashboard.pbix
```

### Batch Analysis Output

The dashboard uses generated CSV data from:

```text
results/batch_resume_jd_scores.csv
```

Since the `results/` folder is ignored in GitHub, users can regenerate the CSV by running:

```bash
python src/batch_analyzer.py
```


## Current Limitations

* The system mainly uses keyword-based matching.
* TF-IDF does not fully understand semantic meaning.
* Skill detection depends on the predefined skill dictionary.
* The ATS score is a simplified approximation and not an actual company ATS system.
* The app currently analyzes one resume and one job description at a time.

## Future Improvements

* Add semantic similarity using sentence embeddings
* Add batch resume-job comparison
* Build a Power BI dashboard using exported CSV data
* Add downloadable PDF analysis reports
* Add more advanced resume tailoring suggestions
* Add support for DOCX resumes
* Add charts for skill categories and score breakdown

## Author

Mridhula Polur
