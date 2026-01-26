import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Car Rental Analytics", layout="wide")

def get_data():
    """Fetch data from the shared SQLite database."""
    conn = sqlite3.connect('car_rental.db')
    
    # Fetch Cars Data
    cars_df = pd.read_sql_query("SELECT * FROM cars", conn)
    
    # Fetch Bookings Data
    bookings_df = pd.read_sql_query("SELECT * FROM bookings", conn)
    
    conn.close()
    return cars_df, bookings_df

# ==========================================
# DASHBOARD LAYOUT
# ==========================================
try:
    cars_df, bookings_df = get_data()

    st.title("🚗 Car Rental System: Executive Dashboard")
    st.markdown("Real-time analytics for fleet management and revenue tracking.")
    
    # --- METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_cars = len(cars_df)
    available_cars = len(cars_df[cars_df['available_now'] == 1])
    rented_cars = total_cars - available_cars
    total_revenue = bookings_df['total_fee'].sum() if not bookings_df.empty else 0
    
    col1.metric("Total Fleet Size", f"{total_cars} Vehicles")
    col2.metric("Available Now", f"{available_cars}", delta=f"{(available_cars/total_cars)*100:.0f}%" if total_cars else "0%")
    col3.metric("Currently Rented", f"{rented_cars}", delta_color="inverse")
    col4.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    st.divider()

    # --- CHARTS ROW 1 ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Fleet Availability Status")
        if not cars_df.empty:
            # Create a Pie Chart for Availability
            status_counts = cars_df['available_now'].map({1: 'Available', 0: 'Rented'}).value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            fig_pie = px.pie(status_counts, values='Count', names='Status', 
                             color='Status',
                             color_discrete_map={'Available':'#00CC96', 'Rented':'#EF553B'},
                             hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No car data available.")

    with c2:
        st.subheader("Revenue by Car Model")
        if not bookings_df.empty and not cars_df.empty:
            # Join Bookings with Cars to get Model names
            merged_df = pd.merge(bookings_df, cars_df, left_on='car_id', right_on='id', suffixes=('_b', '_c'))
            revenue_by_car = merged_df.groupby('model')['total_fee'].sum().reset_index()
            
            fig_bar = px.bar(revenue_by_car, x='model', y='total_fee',
                             labels={'total_fee': 'Revenue ($)', 'model': 'Car Model'},
                             color='total_fee',
                             color_continuous_scale='Bluyl')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No booking data available yet.")

    # --- DATA TABLE ROW ---
    st.subheader("Recent Booking Transactions")
    if not bookings_df.empty:
        # Show the last 5 bookings
        st.dataframe(bookings_df.tail(5).iloc[::-1], use_container_width=True)
    else:
        st.info("No transactions found in the database.")

except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.warning("Please ensure 'main.py' has been run at least once to generate 'car_rental.db'.")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Admin Actions")
    st.write("This dashboard refreshes automatically when the database changes.")
    if st.button("Refresh Data"):
        st.rerun()
