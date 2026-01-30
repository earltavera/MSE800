import sqlite3
import streamlit as st
from abc import ABC, abstractmethod
from datetime import datetime

# ==========================================
# 1. SINGLETON DATABASE MANAGER
# ==========================================
class DatabaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.connection = sqlite3.connect('car_rental.db')
            cls._instance.create_tables()
        return cls._instance

    def get_connection(self):
        return self.connection

    def create_tables(self):
        cursor = self.connection.cursor()
        
        # Table: Users
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE,
                            password TEXT,
                            role TEXT)''')

        # Table: Cars
        cursor.execute('''CREATE TABLE IF NOT EXISTS cars (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            make TEXT,
                            model TEXT,
                            year INTEGER,
                            mileage INTEGER,
                            available_now INTEGER DEFAULT 1,
                            min_rent_period INTEGER,
                            max_rent_period INTEGER,
                            daily_rate REAL)''')

        # Table: Bookings
        cursor.execute('''CREATE TABLE IF NOT EXISTS bookings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            customer_id INTEGER,
                            car_id INTEGER,
                            start_date TEXT,
                            end_date TEXT,
                            total_fee REAL,
                            status TEXT DEFAULT 'Pending',
                            FOREIGN KEY(customer_id) REFERENCES users(id),
                            FOREIGN KEY(car_id) REFERENCES cars(id))''')
        self.connection.commit()

# ==========================================
# 2. ABSTRACT USER & FACTORY PATTERN
# ==========================================
class User(ABC):
    def __init__(self, user_id, username):
        self.user_id = user_id
        self.username = username
        self.db = DatabaseManager().get_connection()

    @abstractmethod
    def menu(self):
        pass

class UserFactory:
    @staticmethod
    def create_user(user_id, username, role):
        if role == 'Admin':
            return Admin(user_id, username)
        elif role == 'Customer':
            return Customer(user_id, username)
        else:
            raise ValueError("Invalid Role")

# ==========================================
# 3. ADMIN CLASS
# ==========================================
class Admin(User):
    def menu(self):
        while True:
            print(f"\n--- ADMIN MENU ({self.username}) ---")
            print("1. Add Car")
            print("2. Update Car Mileage")
            print("3. Delete Car")
            print("4. Manage Bookings (Approve/Reject)")
            print("5. View Fleet Status (Rented vs Available)")
            print("6. Logout")
            choice = input("Select option: ")

            if choice == '1': self.add_car()
            elif choice == '2': self.update_car()
            elif choice == '3': self.delete_car()
            elif choice == '4': self.manage_bookings()
            elif choice == '5': self.view_fleet_status()
            elif choice == '6': break
            else: print("Invalid choice.")

    def view_fleet_status(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT id, make, model, available_now FROM cars")
        cars = cursor.fetchall()
        
        print("\n--- FLEET STATUS ---")
        print(f"{'ID':<5} {'Car':<20} {'Status':<15}")
        print("-" * 55)
        
        for car in cars:
            status = "AVAILABLE" if car[3] == 1 else "RENTED"
            if status == "RENTED":
                cursor.execute("""
                    SELECT u.username, b.end_date 
                    FROM bookings b 
                    JOIN users u ON b.customer_id = u.id 
                    WHERE b.car_id = ? AND b.status = 'Approved'
                """, (car[0],))
                rental_info = cursor.fetchone()
                if rental_info:
                    status = f"Rented by {rental_info[0]} (until {rental_info[1]})"
            
            print(f"{car[0]:<5} {car[1] + ' ' + car[2]:<20} {status:<15}")

    def add_car(self):
        print("\n--- ADD NEW CAR ---")
        make = input("Make: ")
        model = input("Model: ")
        
        try:
            year = int(input("Year: ").replace(',', ''))
            mileage = int(input("Mileage: ").replace(',', ''))
            min_rent = int(input("Min Rent Days: ").replace(',', ''))
            max_rent = int(input("Max Rent Days: ").replace(',', ''))
            daily_rate = float(input("Daily Rate ($): ").replace(',', '').replace('$', ''))
        except ValueError:
            print("Error: Please enter valid numbers only.")
            return

        cursor = self.db.cursor()
        cursor.execute('''INSERT INTO cars (make, model, year, mileage, min_rent_period, max_rent_period, daily_rate)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                       (make, model, year, mileage, min_rent, max_rent, daily_rate))
        self.db.commit()
        print(f"Success! {year} {make} {model} added to fleet.")

    def update_car(self):
        car_id = input("Enter Car ID to update: ")
        try:
            new_mileage = int(input("Enter new mileage: ").replace(',', ''))
            cursor = self.db.cursor()
            cursor.execute("UPDATE cars SET mileage = ? WHERE id = ?", (new_mileage, car_id))
            self.db.commit()
            print("Car updated.")
        except ValueError:
            print("Invalid mileage format.")

    def delete_car(self):
        car_id = input("Enter Car ID to delete: ")
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM cars WHERE id = ?", (car_id,))
        self.db.commit()
        print("Car deleted.")

    def manage_bookings(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM bookings WHERE status = 'Pending'")
        bookings = cursor.fetchall()
        
        if not bookings:
            print("No pending bookings.")
            return

        print("\n--- Pending Bookings ---")
        for b in bookings:
            print(f"Booking ID: {b[0]} | Car ID: {b[2]} | Dates: {b[3]} to {b[4]} | Fee: ${b[5]}")
        
        b_id = input("Enter Booking ID to process: ")
        action = input("Approve (A) or Reject (R)? ").upper()
        
        new_status = 'Approved' if action == 'A' else 'Rejected'
        cursor.execute("UPDATE bookings SET status = ? WHERE id = ?", (new_status, b_id))
        
        if new_status == 'Approved':
            cursor.execute("SELECT car_id FROM bookings WHERE id = ?", (b_id,))
            car_id = cursor.fetchone()[0]
            cursor.execute("UPDATE cars SET available_now = 0 WHERE id = ?", (car_id,))

        self.db.commit()
        print(f"Booking {new_status}!")

# ==========================================
# 4. CUSTOMER CLASS 
# ==========================================
class Customer(User):
    def menu(self):
        while True:
            print(f"\n--- CUSTOMER MENU ({self.username}) ---")
            print("1. View Available Cars")
            print("2. Book a Car")
            print("3. View My Bookings")
            print("4. Logout")
            choice = input("Select option: ")

            if choice == '1': self.view_cars()
            elif choice == '2': self.book_car()
            elif choice == '3': self.view_history()
            elif choice == '4': break

    def view_cars(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT id, make, model, daily_rate, min_rent_period, max_rent_period FROM cars WHERE available_now = 1")
        cars = cursor.fetchall()
        print("\n--- Available Cars ---")
        print(f"{'ID':<5} {'Car':<20} {'Rate/Day':<10} {'Limits (Days)':<15}")
        for c in cars:
            print(f"{c[0]:<5} {c[1] + ' ' + c[2]:<20} ${c[3]:<10} {c[4]}-{c[5]}")

    def calculate_rental_fee(self, start_date, end_date, daily_rate):
        """
        Interaction component: Calculates fee based on duration.
        """
        days = (end_date - start_date).days
        return days, (days * daily_rate)

    def book_car(self):
        self.view_cars()
        car_id = input("\nEnter Car ID to book: ")
        
        start_str = input("Start Date (DD-MM-YYYY): ")
        end_str = input("End Date (DD-MM-YYYY): ")

        try:
            start_date = datetime.strptime(start_str, "%d-%m-%Y")
            end_date = datetime.strptime(end_str, "%d-%m-%Y")

            cursor = self.db.cursor()
            cursor.execute("SELECT daily_rate, min_rent_period, max_rent_period, available_now FROM cars WHERE id = ?", (car_id,))
            car = cursor.fetchone()
            
            if not car:
                print("Car not found.")
                return

            rate, min_days, max_days, is_available = car
            
            # Use the calculated component
            days, total_fee = self.calculate_rental_fee(start_date, end_date, rate)

            if days <= 0:
                print("Error: End date must be after start date.")
                return

            if is_available == 0:
                print("Error: This car is currently rented out.")
                return

            if days < min_days or days > max_days:
                print(f"Error: Rental duration must be between {min_days} and {max_days} days.")
                return

            print(f"Total Fee for {days} days: ${total_fee:.2f}")
            confirm = input("Confirm booking? (y/n): ")

            if confirm.lower() == 'y':
                cursor.execute('''INSERT INTO bookings (customer_id, car_id, start_date, end_date, total_fee) 
                                  VALUES (?, ?, ?, ?, ?)''', 
                               (self.user_id, car_id, start_str, end_str, total_fee))
                self.db.commit()
                print("Booking request sent! Waiting for Admin approval.")

        except ValueError:
            print("Invalid date format. Please use DD-MM-YYYY (e.g., 27-01-2026).")

    def view_history(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM bookings WHERE customer_id = ?", (self.user_id,))
        bookings = cursor.fetchall()
        print("\n--- My Bookings ---")
        for b in bookings:
            print(f"Car ID: {b[2]} | Dates: {b[3]} to {b[4]} | Status: {b[6]} | Fee: ${b[5]}")

# ==========================================
# 5. MAIN SYSTEM
# ==========================================
def main():
    db_manager = DatabaseManager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    while True:
        print("\n=== CAR RENTAL SYSTEM ===")
        print("1. Login")
        print("2. Register")
        print("3. Exit")
        choice = input("Select: ")

        if choice == '1':
            username = input("Username: ")
            password = input("Password: ")
            cursor.execute("SELECT id, role FROM users WHERE username = ? AND password = ?", (username, password))
            user_data = cursor.fetchone()

            if user_data:
                user = UserFactory.create_user(user_data[0], username, user_data[1])
                user.menu()
            else:
                print("Login failed.")

        elif choice == '2':
            username = input("New Username: ")
            password = input("New Password: ")
            role = input("Role (Admin/Customer): ").capitalize()
            
            if role not in ['Admin', 'Customer']:
                print("Invalid role. Must be 'Admin' or 'Customer'.")
                continue

            try:
                cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
                conn.commit()
                print("Registration successful! Please login.")
            except sqlite3.IntegrityError:
                print("Username already exists.")

        elif choice == '3':
            print("Thank you and Goodbye!")
            break
            

if __name__ == "__main__":
    main()
