# Earl Tavera
# Week 3-Activity 1


class ReadFiles:
    """
    A simple class to read files and count specific characters.
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
                return content
        except Exception as e:
            print(f"Error: {e}")
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
            print(f"Error: {e}")
            return 0

if __name__ == "__main__":
 
    reader = ReadFiles("/Users/earltavera/Downloads/demo_file.txt")
    reader.read_and_output()
    reader.count_asterisks('*')
