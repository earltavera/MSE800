import streamlit as st
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

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

    def create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE,
                            password TEXT,
                            role TEXT)''')
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

# Initialize DB
db_manager = DatabaseManager()
conn = db_manager.connection

# ==========================================
# 2. APP LOGIC / HELPERS
# ==========================================

def login_user(username, password):
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone()

def register_user(username, password, role):
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# ==========================================
# 3. STREAMLIT UI COMPONENTS
# ==========================================

def admin_dashboard(user_id, username):
    st.sidebar.title(f"Welcome, Admin {username}")
    menu = st.sidebar.radio("Navigation", ["Fleet Status", "Add Car", "Manage Bookings", "Update/Delete Car"])

    cursor = conn.cursor()

    if menu == "Fleet Status":
        st.header("Current Fleet Status")
        df = pd.read_sql_query("SELECT id, make, model, year, mileage, available_now FROM cars", conn)
        df['Status'] = df['available_now'].apply(lambda x: "Available" if x == 1 else "Rented")
        st.dataframe(df[['id', 'make', 'model', 'year', 'mileage', 'Status']], use_container_width=True)

    elif menu == "Add Car":
        st.header("Add a New Vehicle")
        with st.form("add_car_form"):
            col1, col2 = st.columns(2)
            make = col1.text_input("Make")
            model = col2.text_input("Model")
            year = col1.number_input("Year", min_value=1900, max_value=2026, value=2024)
            mileage = col2.number_input("Mileage", min_value=0)
            min_r = col1.number_input("Min Rent (Days)", min_value=1)
            max_r = col2.number_input("Max Rent (Days)", min_value=1)
            rate = st.number_input("Daily Rate ($)", min_value=0.0)
            
            if st.form_submit_button("Add Car"):
                cursor.execute("INSERT INTO cars (make, model, year, mileage, min_rent_period, max_rent_period, daily_rate) VALUES (?,?,?,?,?,?,?)",
                               (make, model, year, mileage, min_r, max_r, rate))
                conn.commit()
                st.success(f"{year} {make} {model} added!")

    elif menu == "Manage Bookings":
        st.header("Pending Approvals")
        pending = pd.read_sql_query("SELECT * FROM bookings WHERE status = 'Pending'", conn)
        if pending.empty:
            st.info("No pending requests.")
        else:
            st.table(pending)
            booking_id = st.number_input("Enter Booking ID to Process", step=1, min_value=0)
            col1, col2 = st.columns(2)
            if col1.button("Approve", type="primary"):
                cursor.execute("UPDATE bookings SET status = 'Approved' WHERE id = ?", (booking_id,))
                cursor.execute("UPDATE cars SET available_now = 0 WHERE id = (SELECT car_id FROM bookings WHERE id = ?)", (booking_id,))
                conn.commit()
                st.rerun()
            if col2.button("Reject"):
                cursor.execute("UPDATE bookings SET status = 'Rejected' WHERE id = ?", (booking_id,))
                conn.commit()
                st.rerun()

def customer_dashboard(user_id, username):
    st.sidebar.title(f"Hello, {username}")
    menu = st.sidebar.radio("Navigation", ["Available Cars", "My Bookings"])
    cursor = conn.cursor()

    if menu == "Available Cars":
        st.header("Rent a Car")
        cars = pd.read_sql_query("SELECT id, make, model, daily_rate, min_rent_period, max_rent_period FROM cars WHERE available_now = 1", conn)
        
        if cars.empty:
            st.warning("No cars currently available.")
        else:
            st.dataframe(cars, use_container_width=True)
            
            with st.expander("Book a Car"):
                car_id = st.number_input("Enter Car ID", step=1, min_value=0)
                d1 = st.date_input("Start Date")
                d2 = st.date_input("End Date")
                
                if st.button("Calculate & Book"):
                    days = (d2 - d1).days
                    car_data = cursor.execute("SELECT daily_rate, min_rent_period, max_rent_period FROM cars WHERE id=?", (car_id,)).fetchone()
                    
                    if car_data:
                        rate, min_p, max_p = car_data
                        if days < min_p or days > max_p:
                            st.error(f"Rental must be between {min_p} and {max_p} days.")
                        else:
                            total = days * rate
                            cursor.execute("INSERT INTO bookings (customer_id, car_id, start_date, end_date, total_fee) VALUES (?,?,?,?,?)",
                                           (user_id, car_id, str(d1), str(d2), total))
                            conn.commit()
                            st.success(f"Request sent! Total estimated fee: ${total:.2f}")
                    else:
                        st.error("Car ID not found.")

    elif menu == "My Bookings":
        st.header("Your Rental History")
        history = pd.read_sql_query(f"SELECT * FROM bookings WHERE customer_id = {user_id}", conn)
        st.dataframe(history, use_container_width=True)

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
def main():
    st.set_page_config(page_title="Car Rental System", layout="wide")

    if 'user' not in st.session_state:
        st.title("🚗 Car Rental Pro")
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("Login"):
                res = login_user(u, p)
                if res:
                    st.session_state['user'] = res
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")
            role = st.selectbox("Role", ["Customer", "Admin"])
            if st.button("Register"):
                if register_user(new_u, new_p, role):
                    st.success("Account created! You can now login.")
                else:
                    st.error("Username already exists.")
    else:
        # Logged In State
        uid, uname, urole = st.session_state['user']
        if st.sidebar.button("Logout"):
            del st.session_state['user']
            st.rerun()

        if urole == 'Admin':
            admin_dashboard(uid, uname)
        else:
            customer_dashboard(uid, uname)

if __name__ == "__main__":
    main()
