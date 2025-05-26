import re
import json
import pypdfium2 as pdfium
import spacy
from difflib import SequenceMatcher
from collections import defaultdict

import tempfile
import random
from collections import Counter

import os, sys
import shutil
import glob
import numpy as np
import pandas as pd

# Load Spacy Model
nlp = spacy.load("en_core_web_sm")

# Parse PDF
def parse_pdf(resume):
    
    # Parse Document
    pdf = pdfium.PdfDocument(resume)
    n_pages = len(pdf)  

    # Loop Through and Extract All Text into a Single String
    text_final = ""
    for i in range(n_pages):
        page = pdf[i]
        textpage = page.get_textpage()
        text_all = textpage.get_text_range()
        text_final += text_all

    text_final_list = text_final.split("\r\n")
    
    return text_final_list

# Define the Taxonomy
eng_taxonomy = {
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "career history",
        "jobs",
        "work history",
    ],
    "education": [
        "education",
        "academic background",
        "educational background",
        "qualifications",
        "academic qualifications",
        "education & professional credentials"
    ],
    "skills": [
        "skills",
        "technical skills",
        "proficiencies",
        "expertise",
        "competencies",
        "skill set",
    ],
    "others": [
        "professional certification"
        "certifications", 
        "certificates",
        "hackathons",
        "awards"
    ]
}

max_len_eng = max(len(s.split()) for lst in eng_taxonomy.values() for s in lst)
flat_list_eng = [(syn.lower(), cat) for cat, syns in eng_taxonomy.items() for syn in syns]

# Normalize Header
def normalize_header(header: str, taxonomy: dict, flat_list: list, threshold: float = 0.6):
    h = header.lower()
    # Direct keyword precedence
    for cat in taxonomy:
        if re.search(rf"\b{cat}\b", h):  
            return cat, 1.0

    # Fallback to fuzzy only if no exact keyword match
    best_cat, best_score = None, 0.0
    for syn, cat in flat_list:
        score = SequenceMatcher(None, h, syn).ratio()
        if score > best_score:
            best_cat, best_score = cat, score

    if best_score >= threshold:
        return best_cat, best_score
    return None, best_score

# Extract the Resume Header, Resume Body and Resume Body Sections
def extract_sections(text_final_list, threshold: float = 0.5):
    resume_body_headers = []
    resume_body_start_idx = None

    for idx, item in enumerate(text_final_list):
        # header heuristic: no '|', no dots, no full stops, no numbers and shorter than max taxonomy
        no_digits_re = re.compile(r'^[^0-9]*$')
        if "|" not in item and "・" not in item and "。" not in item and no_digits_re.match(item) and len(item.split()) <= max_len_eng:
            cat, score = normalize_header(item, eng_taxonomy, flat_list_eng, threshold)
            if score > threshold:
                if resume_body_start_idx is None:
                    resume_body_start_idx = idx
                resume_body_headers.append((idx, item, cat or "others"))

    # if we never found a header, put body at end
    if resume_body_start_idx is None:
        resume_body_start_idx = len(text_final_list)

    resume_header = text_final_list[:resume_body_start_idx]
    resume_body   = text_final_list[resume_body_start_idx:]
    return resume_header, resume_body, resume_body_headers

# Fallback: Extract Resume Using LLM
def extract_sections_llm(client, text_final_list):
    context_prompt = f"""

        You are an expert resume section extractor.

        Your task is as follows:
        1. Read the resume carefully. Each string represents a line.
        2. Extract the following:
            - resume_header: This variable should contain all lines that constitute the introductory part of the resume. This typically includes the candidate's name, contact information (phone, email, LinkedIn, portfolio link), and sometimes a brief summary or objective statement if present at the very beginning.
            - resume_body: This variable should contain all lines that make up the main content of the resume, excluding the header. This includes sections like "Experience," "Education," "Skills," "Projects," "Awards," etc., along with their corresponding details.
            - resume_body_headers: This variable should contain only the main section titles found within the resume_body. These are typically headings that introduce major sections of the resume (e.g., "EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS").
        3. Return your answer as a JSON array of objects with three keys
            - "resume_header"
            - "resume_body"
            - "resume_body_headers"
            
        Input:
        {text_final_list}
        
        Output:
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
         contents=context_prompt,
        config={
            'response_mime_type': 'application/json',
        },
    )

    output = response.text
    
    try:
        parsed_output = json.loads(output)
        
        # Check if the parsed output is a list containing a single dictionary
        if isinstance(parsed_output, list) and len(parsed_output) == 1 and isinstance(parsed_output[0], dict):
            final_dict = parsed_output[0] # Extract the dictionary from the list
        elif isinstance(parsed_output, dict):
            final_dict = parsed_output # It's already the dictionary
        else:
            raise ValueError(
                f"Model returned unexpected JSON structure. Expected a dictionary or a list containing one dictionary, got: {type(parsed_output)}"
            )

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic JSON string received:\n{output}")
        raise ValueError("Model did not return valid JSON despite instructions.") from e

    return final_dict.get("resume_header", []), final_dict.get("resume_body", []), final_dict.get("resume_body_headers", [])

def extract_name(lines):
    
    has_digit = re.compile(r"\d")
    
    # Pass 1: NER
    for line in lines:
        clean = line.strip()
        if has_digit.search(clean):
            continue
        doc = nlp(clean)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text

    # Pass 2: fallback on formatting
    for line in lines:
        clean = line.strip()
        if "," in clean or has_digit.search(clean):
            continue
        if clean == clean.upper() or clean.istitle():
            return clean

    return ""

def extract_location(lines):
        
    # Pass 1: NER
    for line in lines:
        clean = line.strip()
        doc = nlp(clean)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                return ent.text

def extract_email(text):
    email_content = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email_content:
        try:
            return email_content[0].split()[0].strip(';')
        except IndexError:
            return None

def extract_mobile(text):
    mobile_no = re.findall(re.compile(
        r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'),
        text)
    if mobile_no:
        temp_number = ''.join(mobile_no[0])
        if len(temp_number) > 10:
            return '+' + temp_number
        else:
            return temp_number

def extract_websites(text):
    pattern = r'\b(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?\b'
    urls = re.findall(pattern, text)
    return urls

def extract_personal_information(resume_header):
    
    # Set to Remember Lines We Extracted
    consumed = set()
    
    # Extract Name
    name = extract_name(resume_header)
    if name:
        consumed.add(name)
    
    # Extract Location
    location = extract_location(resume_header)
    if location:
        consumed.add(location)
    
    # Extrace Only One Email and Mobile
    email = mobile = None
    for item in resume_header:
        if not email and extract_email(item):
            email = extract_email(item)
            consumed.add(item)
        if not mobile and extract_mobile(item):
            mobile = extract_mobile(item)
            consumed.add(item)
        if email and mobile:
            break
    
    # Extract Website and Remove Email Address
    websites = []
    email_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com', '.edu']

    for item in resume_header:
        site = extract_websites(item)
        if site:
            # filter out “websites” that are really just email-domains
            if isinstance(site, list):
                legit = [w for w in site if not any(d in w for d in email_domains)]
                if legit:
                    websites.extend(legit)
                    consumed.add(item)
            else:
                if not any(d in site for d in email_domains):
                    websites.append(site)
                    consumed.add(item)
    
    # Compile Consumed and Remove from Others
    others_list = [
        item for item in resume_header
        if item not in consumed and item.strip()  # drop blank lines
    ]
    others = ", ".join(others_list)
    
    # Compile as JSON
    extracted_resume_intro = [
        {
            "name": name,
            "email": email,
            "mobile": mobile,
            "websites": websites, 
            "location": location,
            "others": others
        }
    ]
    
    return extracted_resume_intro
    
# Extract Resume Body
def extract_resume_body(text_final_list, resume_body, resume_body_headers):
    sections_by_cat = defaultdict(list)
    extended = resume_body_headers + [(len(text_final_list), None, None)]

    for (start, name, cat), (next_start, _, _) in zip(resume_body_headers, extended[1:]):
        end = next_start - 1
        sections_by_cat[cat].append((start, end, name))

    experience_list  = [(s, e) for s, e, _ in sections_by_cat['experience']]
    education_list   = [(s, e) for s, e, _ in sections_by_cat['education']]
    skills_list      = [(s, e) for s, e, _ in sections_by_cat['skills']]
    others_list      = [(s, e) for s, e, _ in sections_by_cat['others']]

    experience_text = []
    for start, end in experience_list:
        experience_text.extend(text_final_list[start : end + 1])
    education_text = []
    for start, end in education_list:
        education_text.extend(text_final_list[start : end + 1])
    skills_text = []
    for start, end in skills_list:
        skills_text.extend(text_final_list[start : end + 1])
    if len(skills_text) == 0:
        skills_text = experience_text
    others_text = []
    for start, end in others_list:
        others_text.extend(text_final_list[start : end + 1])

    extracted_resume_body = [
        {
           "criteria": "Education", 
           "requirement": education_text
        },
        {
           "criteria": "Experience", 
           "requirement": experience_text
        },
        {
           "criteria": "Skills", 
           "requirement": skills_text
        },
        {
           "criteria": "Others", 
           "requirement": others_text
        },
    ] 
    
    return extracted_resume_body
