
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
            cls._instance.connection = sqlite3.connect('car_rental.db', check_same_thread=False)
            cls._instance.create_tables()
        return cls._instance

    def get_connection(self):
        return self.connection

    def create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE, password TEXT, role TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS cars (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, make TEXT, model TEXT,
                            year INTEGER, mileage INTEGER, available_now INTEGER DEFAULT 1,
                            min_rent_period INTEGER, max_rent_period INTEGER, daily_rate REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS bookings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER, car_id INTEGER,
                            start_date TEXT, end_date TEXT, total_fee REAL, status TEXT DEFAULT 'Pending',
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
    def dashboard(self):
        pass

class UserFactory:
    @staticmethod
    def create_user(user_id, username, role):
        if role == 'Admin': return Admin(user_id, username)
        elif role == 'Customer': return Customer(user_id, username)
        raise ValueError("Invalid Role")

# ==========================================
# 3. ADMIN CLASS
# ==========================================
class Admin(User):
    def dashboard(self):
        st.title(f"🛠️ Admin Dashboard: {self.username}")
        choice = st.sidebar.selectbox("Management", ["View Fleet Status", "Add New Car", "Manage Bookings"])

        if choice == "View Fleet Status":
            self.view_fleet_status()
        elif choice == "Add New Car":
            self.add_car()
        elif choice == "Manage Bookings":
            self.manage_bookings()

    def view_fleet_status(self):
        st.subheader("Current Fleet Inventory")
        cursor = self.db.cursor()
        cursor.execute("SELECT id, make, model, year, mileage, available_now FROM cars")
        cars = cursor.fetchall()
        st.table([{"ID": c[0], "Make": c[1], "Model": c[2], "Status": "Available" if c[5] == 1 else "Rented"} for c in cars])

    def add_car(self):
        with st.form("add_car_form"):
            make = st.text_input("Make")
            model = st.text_input("Model")
            year = st.number_input("Year", min_value=1900, max_value=2026, value=2024)
            rate = st.number_input("Daily Rate ($)", min_value=0.0)
            if st.form_submit_button("Add to Fleet"):
                cursor = self.db.cursor()
                cursor.execute("INSERT INTO cars (make, model, year, daily_rate, available_now) VALUES (?,?,?,?,1)", (make, model, year, rate))
                self.db.commit()
                st.success(f"{make} {model} added successfully!")

    def manage_bookings(self):
        st.subheader("Pending Approval Requests")
        cursor = self.db.cursor()
        cursor.execute("SELECT b.id, u.username, b.car_id, b.total_fee FROM bookings b JOIN users u ON b.customer_id = u.id WHERE b.status = 'Pending'")
        pending = cursor.fetchall()
        if not pending:
            st.info("No pending bookings.")
            return
        for b in pending:
            col1, col2 = st.columns([3, 1])
            col1.write(f"Booking #{b[0]} | User: {b[1]} | Fee: ${b[3]}")
            if col2.button(f"Approve #{b[0]}", key=f"app_{b[0]}"):
                cursor.execute("UPDATE bookings SET status = 'Approved' WHERE id = ?", (b[0],))
                cursor.execute("UPDATE cars SET available_now = 0 WHERE id = (SELECT car_id FROM bookings WHERE id = ?)", (b[0],))
                self.db.commit()
                st.rerun()

# ==========================================
# 4. CUSTOMER CLASS 
# ==========================================
class Customer(User):
    def dashboard(self):
        st.title(f"🚗 Customer Dashboard: {self.username}")
        page = st.sidebar.radio("Navigation", ["Browse Cars", "My Booking History"])

        if page == "Browse Cars":
            self.book_car_ui()
        elif page == "My Booking History":
            self.view_history()

    def book_car_ui(self):
        st.subheader("Available Vehicles")
        cursor = self.db.cursor()
        cursor.execute("SELECT id, make, model, daily_rate FROM cars WHERE available_now = 1")
        available_cars = cursor.fetchall()
        
        if not available_cars:
            st.warning("No cars available at the moment.")
            return

        car_options = {f"{c[1]} {c[2]} (${c[3]}/day)": c[0] for c in available_cars}
        selected_car = st.selectbox("Select a Car", list(car_options.keys()))
        
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Pick-up Date")
        end_date = col2.date_input("Return Date")

        if start_date and end_date:
            days = (end_date - start_date).days
            if days > 0:
                # Retrieve daily rate for calculation
                cursor.execute("SELECT daily_rate FROM cars WHERE id = ?", (car_options[selected_car],))
                rate = cursor.fetchone()[0]
                total_fee = days * rate
                st.info(f"Rental Duration: {days} days | Total Fee: **${total_fee:.2f}**")
                
                if st.button("Submit Booking Request"):
                    cursor.execute("INSERT INTO bookings (customer_id, car_id, start_date, end_date, total_fee) VALUES (?,?,?,?,?)",
                                   (self.user_id, car_options[selected_car], start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y"), total_fee))
                    self.db.commit()
                    st.success("Request sent to Admin for approval!")
            else:
                st.error("Error: Return date must be after pick-up date.")

    def view_history(self):
        st.subheader("Your Recent Bookings")
        cursor = self.db.cursor()
        cursor.execute("SELECT car_id, start_date, end_date, total_fee, status FROM bookings WHERE customer_id = ?", (self.user_id,))
        history = cursor.fetchall()
        if history:
            st.dataframe([{"Car ID": h[0], "Start": h[1], "End": h[2], "Fee": f"${h[3]}", "Status": h[4]} for h in history])
        else:
            st.info("You have no booking history.")

# ==========================================
# 5. MAIN SYSTEM (STREAMLIT UI)
# ==========================================
def main():
    st.set_page_config(page_title="Car Rental WebApp", page_icon="🚘")
    db_manager = DatabaseManager()

    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        st.sidebar.header("Authentication")
        mode = st.sidebar.selectbox("Select Action", ["Login", "Register"])
        
        with st.container():
            st.title("Welcome to the Car Rental Hub")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if mode == "Login" and st.button("Login"):
                cursor = db_manager.get_connection().cursor()
                cursor.execute("SELECT id, role FROM users WHERE username = ? AND password = ?", (username, password))
                data = cursor.fetchone()
                if data:
                    st.session_state.user = UserFactory.create_user(data[0], username, data[1])
                    st.rerun()
                else: st.error("Invalid Username or Password")

            if mode == "Register":
                role = st.selectbox("Role", ["Customer", "Admin"])
                if st.button("Create Account"):
                    try:
                        cursor = db_manager.get_connection().cursor()
                        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
                        db_manager.get_connection().commit()
                        st.success("Account created! Please switch to Login.")
                    except: st.error("Username already taken.")
    else:
        # Logout logic in sidebar
        if st.sidebar.button("Log Out"):
            st.session_state.user = None
            st.rerun()
        # Direct user to their dashboard
        st.session_state.user.dashboard()

if __name__ == "__main__":
    main()
