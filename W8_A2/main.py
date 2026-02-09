import os
import sys
from src.parser import PDFParser
from src.reviewer import CVReviewer

def main():
    # 1. Configuration
    api_key = os.getenv("GEMINI_API_KEY")
    cv_path = "my_cv.pdf"  # This will be mounted via Docker volume

    # 2. Validation
    if not os.path.exists(cv_path):
        print(f"Error: '{cv_path}' not found. Did you mount the volume?")
        sys.exit(1)

    print(f"--- Processing {cv_path} ---")

    try:
        # 3. Parsing
        parser = PDFParser()
        cv_text = parser.extract_text(cv_path)
        print("✓ Text extracted successfully.")

        # 4. Analysis
        reviewer = CVReviewer(api_key)
        print("✓ Sending to Gemini API...")
        feedback = reviewer.analyze(cv_text)

        # 5. Output
        print("\n" + "="*30)
        print("   GEMINI FEEDBACK REPORT   ")
        print("="*30 + "\n")
        print(feedback)

    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    main()