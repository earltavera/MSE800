import sqlite3
import streamlit as st
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime

# ==========================================
# 1. SINGLETON DATABASE MANAGER (CACHED)
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
                            username TEXT UNIQUE,
                            password TEXT,
                            role TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS cars (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            make TEXT, model TEXT, year INTEGER,
                            mileage INTEGER, available_now INTEGER DEFAULT 1,
                            min_rent_period INTEGER, max_rent_period INTEGER,
                            daily_rate REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS bookings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            customer_id INTEGER, car_id INTEGER,
                            start_date TEXT, end_date TEXT,
                            total_fee REAL, status TEXT DEFAULT 'Pending',
                            FOREIGN KEY(customer_id) REFERENCES users(id),
                            FOREIGN KEY(car_id) REFERENCES cars(id))''')
        self.connection.commit()

# ==========================================
# 2. USER CLASSES (ADAPTED FOR UI)
# ==========================================
class User(ABC):
    def __init__(self, user_id, username):
        self.user_id = user_id
        self.username = username
        self.db = DatabaseManager().get_connection()

    @abstractmethod
    def render_ui(self):
        pass

class Admin(User):
    def render_ui(self):
        st.title(f"Admin Dashboard: {self.username}")
        menu = st.sidebar.selectbox("Menu", ["Fleet Status", "Add Car", "Manage Bookings"])

        if menu == "Fleet Status":
            self.view_fleet()
        elif menu == "Add Car":
            self.add_car_ui()
        elif menu == "Manage Bookings":
            self.manage_bookings_ui()

    def view_fleet(self):
        st.subheader("Current Fleet")
        df = pd.read_sql_query("SELECT * FROM cars", self.db)
        st.dataframe(df, use_container_width=True)
        
        car_to_del = st.number_input("Enter Car ID to delete", min_value=1, step=1)
        if st.button("Delete Car"):
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM cars WHERE id = ?", (car_to_del,))
            self.db.commit()
            st.rerun()

    def add_car_ui(self):
        st.subheader("Add New Vehicle")
        with st.form("add_car_form"):
            make = st.text_input("Make")
            model = st.text_input("Model")
            year = st.number_input("Year", min_value=1900, max_value=2026, value=2024)
            mileage = st.number_input("Mileage", min_value=0)
            min_d = st.number_input("Min Rent Days", min_value=1)
            max_d = st.number_input("Max Rent Days", min_value=1)
            rate = st.number_input("Daily Rate ($)", min_value=0.0)
            if st.form_submit_button("Add to Fleet"):
                cursor = self.db.cursor()
                cursor.execute("INSERT INTO cars (make, model, year, mileage, min_rent_period, max_rent_period, daily_rate) VALUES (?,?,?,?,?,?,?)",
                               (make, model, year, mileage, min_d, max_d, rate))
                self.db.commit()
                st.success(f"{make} {model} added!")

    def manage_bookings_ui(self):
        st.subheader("Pending Requests")
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM bookings WHERE status = 'Pending'")
        rows = cursor.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=['ID', 'User ID', 'Car ID', 'Start', 'End', 'Fee', 'Status'])
            st.table(df)
            b_id = st.number_input("Booking ID to Action", min_value=1, step=1)
            col1, col2 = st.columns(2)
            if col1.button("Approve"):
                cursor.execute("UPDATE bookings SET status = 'Approved' WHERE id = ?", (b_id,))
                cursor.execute("UPDATE cars SET available_now = 0 WHERE id = (SELECT car_id FROM bookings WHERE id = ?)", (b_id,))
                self.db.commit()
                st.rerun()
            if col2.button("Reject", type="primary"):
                cursor.execute("UPDATE bookings SET status = 'Rejected' WHERE id = ?", (b_id,))
                self.db.commit()
                st.rerun()
        else:
            st.write("No pending bookings.")

class Customer(User):
    def render_ui(self):
        st.title(f"Customer Portal: {self.username}")
        menu = st.sidebar.radio("Navigation", ["Browse Cars", "Book Now", "My History"])

        if menu == "Browse Cars":
            self.browse()
        elif menu == "Book Now":
            self.book_ui()
        elif menu == "My History":
            self.history()

    def browse(self):
        df = pd.read_sql_query("SELECT id, make, model, daily_rate FROM cars WHERE available_now = 1", self.db)
        st.dataframe(df, use_container_width=True)

    def book_ui(self):
        st.subheader("Request a Rental")
        car_id = st.number_input("Car ID", min_value=1)
        sd = st.date_input("Start Date")
        ed = st.date_input("End Date")
        if st.button("Submit Request"):
            days = (ed - sd).days
            cursor = self.db.cursor()
            cursor.execute("SELECT daily_rate, min_rent_period, max_rent_period FROM cars WHERE id = ?", (car_id,))
            car = cursor.fetchone()
            if car and car[1] <= days <= car[2]:
                fee = days * car[0]
                cursor.execute("INSERT INTO bookings (customer_id, car_id, start_date, end_date, total_fee) VALUES (?,?,?,?,?)",
                               (self.user_id, car_id, str(sd), str(ed), fee))
                self.db.commit()
                st.success(f"Booking submitted! Total estimated: ${fee}")
            else:
                st.error("Invalid duration or Car ID.")

    def history(self):
        df = pd.read_sql_query(f"SELECT * FROM bookings WHERE customer_id = {self.user_id}", self.db)
        st.write(df)

class UserFactory:
    @staticmethod
    def create_user(user_id, username, role):
        return Admin(user_id, username) if role == 'Admin' else Customer(user_id, username)

# ==========================================
# 3. MAIN APP & STATE MANAGEMENT
# ==========================================
def main():
    st.set_page_config(page_title="Car Rental Pro", layout="wide")
    db = DatabaseManager()

    if 'user_obj' not in st.session_state:
        st.sidebar.title("Login / Register")
        mode = st.sidebar.selectbox("Action", ["Login", "Register"])
        user_in = st.sidebar.text_input("Username")
        pass_in = st.sidebar.text_input("Password", type="password")

        if mode == "Login":
            if st.sidebar.button("Login"):
                cursor = db.get_connection().cursor()
                cursor.execute("SELECT id, role FROM users WHERE username = ? AND password = ?", (user_in, pass_in))
                data = cursor.fetchone()
                if data:
                    st.session_state.user_obj = UserFactory.create_user(data[0], user_in, data[1])
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")
        else:
            role_in = st.sidebar.selectbox("Role", ["Customer", "Admin"])
            if st.sidebar.button("Register"):
                try:
                    cursor = db.get_connection().cursor()
                    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (user_in, pass_in, role_in))
                    db.get_connection().commit()
                    st.sidebar.success("Account created! Please login.")
                except:
                    st.sidebar.error("Username taken")
    else:
        if st.sidebar.button("Logout"):
            del st.session_state.user_obj
            st.rerun()
        st.session_state.user_obj.render_ui()

if __name__ == "__main__":
    main()
