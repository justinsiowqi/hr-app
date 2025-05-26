import os
from typing import List, Dict

# Langchain and Groq imports
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the Groq client using environment variable
client = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

def job_description_agent(basic_description) -> str:
    prompt = (
        f"""
        Basic Description: {basic_description}

        Your task:
        1.  Craft a compelling and detailed job description (approx. 300-500 words).
        2.  Highlight what makes working at this startup unique, focusing on impact, growth, and the cultural aspects mentioned.
        3.  Clearly list key responsibilities and qualifications.
        4.  Include a section on "Why Join Us?" that emphasizes the startup DNA.
        5.  Suggest 3-5 key "soft skills" that would indicate a strong cultural fit for this role and startup.
        """
    )
    messages = [
        SystemMessage(content="You are an expert Job Architect for startups."),
        HumanMessage(content=prompt)
    ]
    response = client.invoke(messages)
    return response.content

def resume_screening_agent(job_description, candidate_resume):
        
    prompt = (
        f"""
        Instructions:
        1. Analyze the Job Description:
            - Identify hard requirements (e.g., skills, certifications, experience).
            - Identify soft requirements (e.g., teamwork, leadership).
            - Note any preferred qualifications (bonus points).

        2. Analyze the Candidate’s Resume:
            - Extract relevant skills, experience, and achievements.
            - Highlight quantifiable accomplishments (e.g., "increased sales by 20%").
            - Note any gaps (missing requirements).

        3. Match & Score (0-100):
            - Compare the candidate’s qualifications against the job requirements.
            - Assign weights to critical requirements (e.g., "5+ years of Python" = 30% of score).
            - Deduct points for missing hard requirements; award bonus points for preferred qualifications.
            
        4. Output Structure:
            - Match Score: Integer from 0 to 100.
            
        Example Output:
        {{
            "Score": 88/100,
            " Strengths":  [
                7 years of Python (matches "5+ years" requirement), 
                Led a team of 5 (matches leadership requirement)
            ]
            "Weaknesses": [
                No AWS certification (listed as preferred),
            ]
            "Recommendation": "Strong fit for the role."
        }}
            
        Job Description: 
        {job_description}
        
        Candidate Resume:
        {candidate_resume}
        """
    )
    messages = [
        SystemMessage(content="You are an expert at evaluating people's capabilities."),
        HumanMessage(content=prompt)
    ]
    response = client.invoke(messages)
    return response.content

def interview_question_agent(job_description, candidate_resume):
        
    prompt = (
        f"""
        Instructions:
        1.  Use the Job Description to identify key competencies and responsibilities.
        2.  Review the resume review to understand the candidate's strengths and weaknesses.
        3.  Ensure that questions related to Work Experience are specific to the candidate's past roles and accomplishments as described in their resume, and relevant to the requirements of the job description.
        4.  Generate 7-10 high-quality interview questions that probe areas of strength, address potential weaknesses, and align with the interviewer's focus.
        5.  Ensure questions are open-ended and encourage detailed responses.
        6.  Avoid generic questions. Make them specific to the context provided, including insights from the resume review.
            
        Example Output:
        1. As a highly motivated problem solver with excellent interpersonal and communication skills, can you describe a situation where you had to communicate complex technical ideas to a non-technical audience? 
        2. How do you stay up-to-date with the latest developments in the field of GenAI technology, and how do you see these technologies evolving in the future?"
        ```
            
        Job Description: 
        {job_description}
        
        Candidate Resume:
        {candidate_resume}
        """
    )
    messages = [
        SystemMessage(content="You are an expert in crafting interview questions. "),
        HumanMessage(content=prompt)
    ]
    response = client.invoke(messages)
    return response.content