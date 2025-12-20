# Can we see the different objects respond differently to the same method call? if yes/ no explain it in short? and what the usage of this concept? 
# Answer: Yes, we can define a common method called move() in the base class and override it in each subclass.
# The usage of this concept is to context the animal hierarchy, polymorphism allows different classes to have a method with the same name but different implementations. 
# When you call that method on an object, Python automatically executes the version specific to that object's class.


# Base Class 
class Animal:
    """The root class for all animals."""
    def __init__(self, name: str):
        self.name = name

    def display_identity(self):
        print(f"I am an animal named {self.name}.")


# --- Intermediate Classes (Level 1 Inheritance) ---
class Mammal(Animal):
    """Inherits from Animal. Adds a general feature attribute."""
    def __init__(self, name: str, feature: str):
        super().__init__(name)  # Pass name to the Animal constructor
        self.feature = feature

class Bird(Animal):
    """Inherits from Animal."""
    def __init__(self, name: str, feature: str):
        super().__init__(name)
        self.feature = feature

class Fish(Animal):
    """Inherits from Animal."""
    def __init__(self, name: str, feature: str):
        super().__init__(name)
        self.feature = feature


# --- Specific Species (Level 2 Inheritance / Multi-level) ---

# Mammal Subclasses
class Dog(Mammal):
    """Inherits name from Animal and feature from Mammal."""
    def walk(self) -> None:
        print(f"{self.name} the Dog (Feature: {self.feature}) is walking.")

class Cat(Mammal):
    def walk(self) -> None:
        print(f"{self.name} the Cat (Feature: {self.feature}) is walking.")

# Bird Subclasses
class Eagle(Bird):
    def fly(self) -> None:
        print(f"{self.name} the Eagle (Feature: {self.feature}) is flying high.")

class Penguin(Bird):
    def swim(self) -> None:
        print(f"{self.name} the Penguin (Feature: {self.feature}) is swimming in cold water.")

# Fish Subclasses
class Salmon(Fish):
    def swim(self) -> None:
        print(f"{self.name} the Salmon (Feature: {self.feature}) is swimming upstream.")

class Shark(Fish):
    def swim(self) -> None:
        print(f"{self.name} the Shark (Feature: {self.feature}) is swimming rapidly.")


# --- Execution and Demonstration ---
if __name__ == "__main__":
    print("--- Animal Kingdom Inheritance Demo ---")

    # Creating instances of different levels
    my_dog = Dog("Bogart", "Fur")
    my_eagle = Eagle("Tweety", "Feathers")
    my_shark = Shark("Ben", "Scales")

    # Demonstrating inherited attributes and unique methods
    my_dog.walk()
    my_eagle.fly()
    my_shark.swim()
    
    # Accessing the root attribute 'name' inherited from the base class
    print(f"\nRoot name check: {my_dog.name}, {my_eagle.name}, {my_shark.name}")
