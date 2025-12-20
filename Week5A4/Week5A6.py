 # Both objects are asked to 'greet()', but they respond differently

class Person:
    def __init__(self, name, address, age):
        # Using a single underscore is a convention for 'protected' attributes
        self._name = name 
        self.address = address
        self.age = age

    def greet(self):
        print(f"Hello, I am a general person named {self._name}.")

class Student(Person): 
    def __init__(self, name, address, age, student_id):
        # Corrected super() call
        super().__init__(name, address, age)
        self.student_id = student_id

    # This overrides the Person.greet() method
    def greet(self):
        print(f"Hi! I'm {self._name} and my student ID is {self.student_id}.")

# Demonstration
person1 = Person("John", "456 Oak St", 40)
student1 = Student("Alice", "123 Main St", 20, "S12345")

# Different objects responding to the same method call
entities = [person1, student1]

for entity in entities:
    entity.greet()

#######################################

# --- Base Class ---
class Person:
    def __init__(self, name, address, age):
        # Using _name as a protected attribute (convention)
        self._name = name
        self.address = address
        self.age = age

    def greet(self):
        """Standard formal greeting for any person."""
        print("Greetings and felicitations from the maestro " + self._name)


# --- Subclass (Inheritance) ---
class Student(Person): 
    def __init__(self, name, address, age, student_id):
        # super() links this Student to the Person constructor
        super().__init__(name, address, age)
        self.student_id = student_id

    def greet(self):
        """Overridden greeting specific to students (Polymorphism)."""
        print(f"Hi! I'm {self._name}. My student ID is {self.student_id}.")


# --- Demonstration of Different Responses ---
if __name__ == "__main__":
    # Create different objects
    person_obj = Person("John", "456 Oak St", 40)
    student_obj = Student("Alice", "123 Main St", 20, "S12345")

    # Both objects are asked to 'greet()', but they respond differently
    print("--- Person Object Response ---")
    person_obj.greet()

    print("\n--- Student Object Response ---")
    student_obj.greet()
