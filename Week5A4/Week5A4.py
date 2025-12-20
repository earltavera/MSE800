# Base Class #
class Person:
    """Represents the root of the hierarchy."""
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def display_info(self):
        print(f"ID: {self.id} | Name: {self.name}")


# Child Class of Person #
class Student(Person):
    """Inherits from Person. Adds student-specific ID."""
    def __init__(self, id, name, student_id):
# Using super() to initialize base Person attributes
        super().__init__(id, name)
        self.student_id = student_id

    def display_info(self):
        super().display_info()
        print(f"Role: Student | Student ID: {self.student_id}")


class Staff(Person):
    """Inherits from Person. Acts as a base for specific staff types."""
    def __init__(self, id, name, staff_id, tax_num):
        super().__init__(id, name)
        self.staff_id = staff_id
        self.tax_num = tax_num

    def display_info(self):
        super().display_info()
        print(f"Staff ID: {self.staff_id} | Tax Number: {self.tax_num}")


# Second Level Inheritance (Multi-level) #
class General(Staff):
    """Inherits from Staff. Represents administrative/support personnel."""
    def __init__(self, id, name, staff_id, tax_num, rate_of_pay):
# Passes info up to Staff, which then passes important information to Person
        super().__init__(id, name, staff_id, tax_num)
        self.rate_of_pay = rate_of_pay

    def display_info(self):
        super().display_info()
        print(f"Category: General Staff | Pay Rate: ${self.rate_of_pay}/hr")


class Academic(Staff):
    """Inherits from Staff. Represents faculty/researchers."""
    def __init__(self, id, name, staff_id, tax_num, publications):
        super().__init__(id, name, staff_id, tax_num)
        self.publications = publications  # List of strings

    def display_info(self):
        super().display_info()
        print(f"Category: Academic | Publications: {', '.join(self.publications)}")


# Demonstration #
if __name__ == "__main__":
    print("--- University Records ---")
    
    # Create a Student
    s1 = Student(1, "Albert Smith", "S10023")
    s1.display_info()
    print()

    # Create General Staff
    g1 = General(2, "Roxy Miller", "ST99", "TAX-882", 45.0)
    g1.display_info()
    print()

    # Create Academic Staff
    a1 = Academic(3, "Dr. Benjz Jones", "ST42", "TAX-115", ["MSE", "Pinoy Pawnstar"])
    a1.display_info()
