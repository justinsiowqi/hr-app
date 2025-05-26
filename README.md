# AI‑Powered HR Assistant

This Streamlit application assists HR professionals in improving their hiring process by leveraging AI agents. It offers modules for:

* **Job Description**: Generate and refine job descriptions using natural language prompts.
* **Resume Parsing**: Extract structured data from candidate resumes for easy review.
* **Candidate Scoring**: Score candidates against the job description to identify top fits.
* **Interview Question**: Automatically generate tailored interview questions.

---

## Features

1. **Job Description Page**

   * Paste or prompt for a job description.
   * Edit and save the generated output.

2. **Resume Parsing Page**

   * Upload or paste candidate resumes.
   * Parses and displays key sections (Education, Experience, Skills, Others).

3. **Candidate Scoring Page**

   * Compares each parsed resume against the job description.
   * Produces a numeric match score (0–100).
   * Displays a summary table of candidate names and scores.

4. **Interview Question Page**

   * Generates interview questions based on the job requirements and candidate profile.

---

## Installation

1. Clone the repository:

   ```bash
   git clone [<REPO_URL>](https://github.com/justinsiowqi/hr-app)
   cd hr-app
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\\Scripts\\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Add the Groq API Key
   
  ```bash
  EXPORT GROQ_API_KEY=<INSERT_API_KEY_HERE>
  ```

2. Run the Streamlit app:

  ```bash
  streamlit run src/app.py
  ```
