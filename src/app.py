import streamlit as st
import pandas as pd

from resume_parser import *
from agents import job_description_agent, resume_screening_agent, interview_question_agent

def job_description_page():
    st.title("Job Description")
    
    if "final_job" not in st.session_state:
        st.session_state["final_job"] = ""
 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter your request")
        prompt = st.text_area(
            label="",
            key="jd_prompt",
            height=400
        )
        generate_disabled = not bool(prompt.strip())
        if st.button("Generate", key="generate_jd", disabled=generate_disabled):
            result = job_description_agent(prompt)
            st.session_state["final_job"] = result

    with col2:
        st.subheader("Generated description")
        edited = st.text_area(
            label="",
            value=st.session_state["final_job"],
            key="generated_jd_area",
            height=400
        )
        st.session_state["final_job"]= edited

def resume_parsing_page():
    st.title("Resume Parsing")
    
    # Upload PDFs
    uploads = st.file_uploader(
        "Upload CVs",
        type="pdf",
        accept_multiple_files=True,
        key="resume_uploads"    
    )
    
    if st.button("Parse & Extract Resumes"):
        if not uploads:
            st.warning("Upload at least one PDF.")
            return
        
        resume_intros = []
        resume_bodies = []
        for uploaded_resume in uploads:
            pdf_bytes = uploaded_resume.read()
            parsed   = parse_pdf(pdf_bytes)
            
            resume_header, resume_body, resume_body_headers = extract_sections(parsed)
                
            # if isinstance(resume_header, list) and len(resume_header) == 0 or \
            #     isinstance(resume_body, list) and len(resume_body) == 0 or \
            #     isinstance(resume_body_headers, list) and len(resume_body_headers) == 0:
            #     resume_header, resume_body, resume_body_headers = extract_sections_llm(client, clean_ascii(parsed))

            intro = extract_personal_information(resume_header)
            resume_intros.append(intro)

            body = extract_resume_body(parsed, resume_body, resume_body_headers)
            resume_bodies.append(body)
        
        # Merge Resume Intro & Body
        intro_rows = [item for sub in resume_intros for item in sub]
        body_rows  = [{ rec['criteria']: "\n".join(rec['requirement']) 
                        for rec in resume } 
                       for resume in resume_bodies]
        intro_df = pd.DataFrame(intro_rows)
        body_df  = pd.DataFrame(body_rows)
        final_resume_df = pd.concat([intro_df, body_df], axis=1)
        
        # Store in Session State
        st.session_state["final_resume"] = final_resume_df
    
    # Grab the Current Dataframe
    cur_df = st.session_state.get("final_resume", pd.DataFrame())
    if cur_df is None or cur_df.empty:
        return  

    st.subheader("Extracted Resumes")
    st.data_editor(cur_df, use_container_width=True, key="resume_editor")
    
def candidate_scoring_page():
    st.title("Candidate Scoring")

    # Score Button
    if st.button("Score Candidates"):
        job = st.session_state.get("final_job", [])
        res = st.session_state.get("final_resume", [])
        
        if job is None or \
            (isinstance(job, pd.DataFrame) and job.empty) or \
            (isinstance(job, list) and len(job) == 0):
            st.warning("Parse job description first.")
            return
        if res is None or \
            (isinstance(res, pd.DataFrame) and res.empty) or \
            (isinstance(res, list) and len(res) == 0):
            st.warning("Parse at least one resume first.")
            return
        
        for idx, row in res.iterrows():
            res_compiled = row["Education"] + "\n" + row["Experience"] + "\n" + row["Skills"] + "\n" + row["Others"]
            json_output = resume_screening_agent(job, res_compiled)
                        
            st.subheader("Candidate Analysis")
            st.write(json_output)
            st.write("==============================")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('\\n', '\n')  
    text = text.replace('\t', ' ')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = text.strip()
    return text

def format_resume_row(row: pd.Series) -> str:
    sections = []

    # Top header info
    header_fields = ["name", "email", "mobile", "websites", "location", "others"]
    for field in header_fields:
        if field in row and row[field] is not None and not (isinstance(row[field], float) and pd.isna(row[field])):
            value = row[field]
            if isinstance(value, (list, tuple)):
                value = ", ".join(map(str, value))
            cleaned = clean_text(str(value))
            if cleaned.strip():
                sections.append(f"**{field.title()}:** {cleaned}")

    # Resume content sections
    content_sections = ["Education", "Experience", "Skills", "Others"]
    for section in content_sections:
        if section in row and pd.notna(row[section]):
            cleaned = clean_text(str(row[section]))
            if cleaned.strip():
                sections.append(f"\n### {section.title()}\n{cleaned}")

    return "\n\n".join(sections)

def interview_question_page():
    st.title("Interview Questions")
    
    col1, col2 = st.columns(2)

    with col1:
        if "final_job" not in st.session_state or not isinstance(st.session_state["final_job"], str):
            st.session_state["final_job"] = ""
            
        st.subheader("Job Description")
        job_description = st.text_area(
            "Job Description",
            height=585,
            placeholder="Paste job description...",
            value=st.session_state["final_job"],
        )

    with col2:
        st.subheader("Candidate Resume")
        final_resume = st.session_state["final_resume"]

        if isinstance(final_resume, pd.DataFrame) and not final_resume.empty and "name" in final_resume.columns:
            candidate_names = final_resume["name"].tolist()
            selected_name = st.selectbox("Select a candidate", candidate_names)

            # Select entire row (as a Series) for the selected candidate
            selected_row = final_resume[final_resume["name"] == selected_name]

            if not selected_row.empty:
                with pd.option_context('display.max_colwidth', None):
                    resume_row = selected_row.iloc[0]
                    row_text = format_resume_row(resume_row)
            else:
                row_text = ""
        else:
            st.selectbox("Select a candidate", options=["No candidates available"], index=0, disabled=True)
            row_text = ""
        
        candidate_resume = st.text_area(
            "Candidate Resume",
            height=500,
            placeholder="Paste candidate resume...",
            value=row_text,
        )
        
    if st.button("Generate Questions"):
        if not job_description or not candidate_resume:
            st.error("Please fill in both the job description and candidate resume.")
        else:
            output = interview_question_agent(final_resume, candidate_resume)
            st.markdown(output)
    
# Page Navifation
PAGES = {
    "Job Description": job_description_page,
    "Resume Parsing": resume_parsing_page, 
    "Candidate Scoring": candidate_scoring_page,
    "Interview Question": interview_question_page,
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
