# Earl Tavera
# Week 3-Activity 2

import os

class ReadFiles:
    """
    A simple class to read files, count specific characters, and append text.
    """
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_and_output(self):
        """Reads and prints the file content."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("=" * 50)
                print(f"Content of: {self.file_path}")
                print("=" * 50)
                print(content)
                print("=" * 50)
                return content
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def count_asterisks(self, char):
        """Counts occurrences of a character."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count(char)
                print(f"Count of '{char}': {count}")
                return count
        except Exception as e:
            print(f"Error counting characters: {e}")
            return 0

    def append_text(self, text):
        """Appends text to the end of the file."""
        try:
            # Open in 'a' (append) mode
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{text}")
            print(f"Successfully appended: '{text}'")
        except Exception as e:
            print(f"Error appending text: {e}")

if __name__ == "__main__":
    # 1. Assign File and directory
    target_file = "/Users/earltavera/Downloads/demo_file.txt"
    
    with open(target_file, "w", encoding="utf-8") as f:
        f.write("This is a sample file with some asterisks *** inside.")

    # 2. Initialize the class
    reader = ReadFiles(target_file)

    # 3. Read original content
    print("--- Initial Read ---")
    reader.read_and_output()

    # 4. Count asterisks
    reader.count_asterisks('*')

    # 5. Append "end of file"
    print("\n--- Appending Text ---")
    reader.append_text("end of file")

    # 6. Read again to verify the append
    print("\n--- Verification Read ---")
    reader.read_and_output()
