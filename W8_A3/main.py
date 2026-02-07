import os
import sys
import pdfplumber
import docx
import google.generativeai as genai

# --- Configuration ---
# 1. Get API Key securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Fallback: Ask user for key if not passed in Docker command
    GEMINI_API_KEY = input("Enter your Google Gemini API Key: ").strip()

if not GEMINI_API_KEY:
    print("Error: No API Key provided.")
    sys.exit(1)

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def list_available_models():
    """Helper to debug model access issues."""
    print("\nChecking available models for your API key...")
    try:
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
        return available
    except Exception as e:
        return [f"Error listing models: {e}"]

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""

def read_text_file(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""

def analyze_match(cv_text, job_description):
    prompt = f"""
    You are an expert technical recruiter. 
    
    JOB DESCRIPTION:
    {job_description}
    
    CANDIDATE CV:
    {cv_text}
    
    TASK:
    1. Give a 'Match Score' (0-100%) based on the requirements.
    2. List 3 specific technical gaps.
    3. List 3 strengths.
    4. Provide specific tailoring advice.
    """

    # Try the standard model first
    model_name = 'gemini-2.5-flash'
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # If the specific model fails, list what IS available to help debug
        print(f"\n❌ Error using model '{model_name}': {e}")
        available_models = list_available_models()
        print(f"ℹ️  Your API Key has access to these models: {available_models}")
        print("👉 Please update the 'model_name' variable in main.py to one of these.\n")
        return "Analysis failed due to model error."

if __name__ == "__main__":
    print("--- CV Analyzer (Gemini Edition) ---")
    
    # 2. Ask for filenames
    cv_path = input("Enter CV filename (default: resume-sample.pdf): ").strip()
    if not cv_path: cv_path = "resume-sample.pdf"

    jd_path = input("Enter Job Description filename (default: prompt.txt): ").strip()
    if not jd_path: jd_path = "prompt.txt"

    # 3. Process
    cv_text = ""
    if cv_path.endswith(".pdf"):
        cv_text = extract_text_from_pdf(cv_path)
    elif cv_path.endswith(".docx"):
        cv_text = extract_text_from_docx(cv_path)
    else:
        print("Unsupported file format. Please use PDF or DOCX.")
        sys.exit(1)

    jd_text = read_text_file(jd_path)

    if cv_text and jd_text:
        print(f"\nAnalyzing {cv_path} against {jd_path}...")
        result = analyze_match(cv_text, jd_text)
        print("\n" + "="*40)
        print("       ANALYSIS REPORT       ")
        print("="*40 + "\n")
        print(result)
    else:
        print("Failed to read one or both files.")