class CourseRecord:
    def __init__(self):
        # Dictionary 1: Stores Student ID -> Name
        self.student_names = {}
        # Dictionary 2: Stores Student ID -> MSE800 Score
        self.student_scores = {}

    def add_student(self, student_id, name, score):
        """Adds a student to both dictionaries."""
        self.student_names[student_id] = name
        self.student_scores[student_id] = score

    def display_records(self):
        """Prints a formatted table of the student records."""
        print(f"{'ID':<10} {'Name':<20} {'MSE800 Score':<12}")
        print("-" * 42)
        
        for s_id, name in self.student_names.items():
            # Retrieve score using the same ID key
            score = self.student_scores.get(s_id, "N/A")
            print(f"{s_id:<10} {name:<20} {score:<12}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Instantiate the class
    mse800_course = CourseRecord()

    # 2. Add five students
    mse800_course.add_student("S101", "Piolo", 88)
    mse800_course.add_student("S102", "Coco", 75)
    mse800_course.add_student("S103", "Richard", 92)
    mse800_course.add_student("S104", "Dindong", 81)
    mse800_course.add_student("S105", "Direk", 67)

    # 3. Display the results
    mse800_course.display_records()
    
    # 4. Show the raw dictionaries to verify the requirement
    print("\n--- Internal Data Structures ---")
    print("Names Dictionary:", mse800_course.student_names)
    print("Scores Dictionary:", mse800_course.student_scores)
