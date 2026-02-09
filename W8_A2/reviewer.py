import os
import google.generativeai as genai

class CVReviewer:
    """Uses Google Gemini API to analyze CV text."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is missing. Please set GEMINI_API_KEY.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze(self, cv_text: str) -> str:
        """
        Sends CV text to Gemini and retrieves actionable feedback.
        """
        prompt = (
            "You are an expert technical recruiter. Analyze the following CV text. "
            "Provide a strict list of 3 actionable improvements to make this CV "
            "stand out for a Senior Engineer role. Be specific, not generic.\n\n"
            f"CV CONTENT:\n{cv_text}"
        )
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {e}"