class CourseRecord:
    def __init__(self):
        # Dictionary 1: ID -> Name
        self.student_names = {}
        # Dictionary 2: ID -> Score
        self.student_scores = {}

    def add_student(self, student_id, name, score):
        """Adds a student to both dictionaries."""
        self.student_names[student_id] = name
        self.student_scores[student_id] = score

    def generate_passed_report(self):
        """
        Combines dictionaries and generates a new dictionary 
        containing only students with score >= 50.
        
        Returns:
            dict: {ID: {'name': Name, 'score': Score}}
        """
        passed_students = {}
        
        # Iterate through the scores dictionary
        for s_id, score in self.student_scores.items():
            if score >= 50:
                # Retrieve the name from the names dictionary
                name = self.student_names[s_id]
                
                # Create a combined entry (Dictionary of details)
                passed_students[s_id] = {'name': name, 'score': score}
        
        return passed_students

    def display_records(self):
        """Helper to print the full raw list (Part 1 requirement)."""
        print(f"{'ID':<10} {'Name':<20} {'MSE800 Score':<12}")
        print("-" * 42)
        for s_id, name in self.student_names.items():
            score = self.student_scores.get(s_id, "N/A")
            print(f"{s_id:<10} {name:<20} {score:<12}")

# --- Main Execution ---
if __name__ == "__main__":
    mse800_course = CourseRecord()

    # Adding students (including your specific examples)
    mse800_course.add_student("S101", "Piolo", 88)
    mse800_course.add_student("S102", "Coco", 75)
    mse800_course.add_student("S103", "Richard", 92)
    mse800_course.add_student("S104", "Dindong", 81)
    mse800_course.add_student("S105", "Direk", 67)
    
    # I added a failing student here to demonstrate that the filter works
    mse800_course.add_student("S106", "Vice", 45) 

    print("--- All Students (Before Filter) ---")
    mse800_course.display_records()

    # Generate the new dictionary for Part 2
    passed_dict = mse800_course.generate_passed_report()

    print("\n--- Part 2: Passed Students Dictionary (Score >= 50) ---")
    # Pretty printing the new dictionary
    for s_id, details in passed_dict.items():
        print(f"ID: {s_id} | Passed: {details}")

    # Verify the internal structure
    print("\n[Raw New Dictionary]:", passed_dict)
