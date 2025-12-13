# EARL TAVERA
# WEEK 3 ACTIVITY 5
# List the full information of all patients who are classified as seniors in the clinic (age > 65 years).
# Display the total number of doctors who specialise in ophthalmology.

# 1. Define the 'Patient' template
class Patient:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Patient: {self.name}, Age: {self.age}"

# 2. Define the 'Doctor' template
class Doctor:
    def __init__(self, name, specialization):
        self.name = name
        self.specialization = specialization

# 3. Define the 'Clinic' to manage the data
class Clinic:
    def __init__(self):
        self.patients = [] # A generic list to hold patients
        self.doctors = []  # A generic list to hold doctors

    # Helper to add people to the clinic
    def add_patient(self, patient):
        self.patients.append(patient)

    def add_doctor(self, doctor):
        self.doctors.append(doctor)

    # REQUIREMENT 1: List all seniors (Age > 65)
    def show_senior_patients(self):
        print("--- Senior Patients (Over 65) ---")
        for p in self.patients:
            if p.age > 65:
                print(p) 

    # REQUIREMENT 2: Count Doctors in Ophthalmology
    def count_ophthalmologists(self):
        count = 0
        for d in self.doctors:
            if d.specialization == "Ophthalmology":
                count += 1
        print(f"\nTotal Ophthalmologists: {count}")

# --- MAIN EXECUTION (Testing the code) ---

# Create the clinic
my_clinic = Clinic()

# Add some dummy Patients
my_clinic.add_patient(Patient("Michael Jordan", 70))   # Senior
my_clinic.add_patient(Patient("Jane Tarzan", 45))
my_clinic.add_patient(Patient("Robert Downy", 82)) # Senior

# Add some dummy Doctors
my_clinic.add_doctor(Doctor("Dr. Benj", "Neurology"))
my_clinic.add_doctor(Doctor("Dr. Roxy", "Ophthalmology"))
my_clinic.add_doctor(Doctor("Dr. Albert", "Ophthalmology"))

# Run the reports
my_clinic.show_senior_patients()
my_clinic.count_ophthalmologists()
