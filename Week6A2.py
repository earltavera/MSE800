import sqlite3

class CourseRecordDB:
    def __init__(self, db_name="mse800_course.db"):
        """Initialize connection and create the table."""
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.initialize_table()

    def initialize_table(self):
        """Creates the Student table. Drops it first to ensure a clean slate for this run."""
        self.cursor.execute("DROP TABLE IF EXISTS Student")
        
        # Create table with the 3 required attributes
        self.cursor.execute("""
            CREATE TABLE Student (
                student_id TEXT PRIMARY KEY,
                student_name TEXT,
                score INTEGER
            )
        """)
        self.conn.commit()

    def add_student(self, student_id, name, score):
        """Inserts a student into the database."""
        try:
            query = "INSERT INTO Student (student_id, student_name, score) VALUES (?, ?, ?)"
            self.cursor.execute(query, (student_id, name, score))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"Error: Student ID {student_id} already exists.")

    def display_top_three(self):
        """Runs an SQL query to find the top 3 students by score."""
        # SQL Query: Order by score descending (High to Low) and take the top 3
        query = "SELECT student_id, student_name, score FROM Student ORDER BY score DESC LIMIT 3"
        self.cursor.execute(query)
        results = self.cursor.fetchall()

        print(f"\n{'ID':<10} {'Name':<20} {'MSE800 Score':<12}")
        print("=" * 42)
        
        for row in results:
            # row is a tuple: (id, name, score)
            print(f"{row[0]:<10} {row[1]:<20} {row[2]:<12}")

    def close_connection(self):
        """Closes the database connection."""
        self.conn.close()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Instantiate the class (creates the DB file automatically)
    course_db = CourseRecordDB()

    # 2. Add the students (Data is now stored in SQLite)
    course_db.add_student("S101", "Piolo", 88)
    course_db.add_student("S102", "Coco", 75)
    course_db.add_student("S103", "Richard", 92)
    course_db.add_student("S104", "Dindong", 81)
    course_db.add_student("S105", "Direk", 67)

    # 3. specific requirement: Run SQL to get Top 3
    print("--- Top 3 Students (SQL Query Result) ---")
    course_db.display_top_three()
    
    # 4. Cleanup
    course_db.close_connection()
