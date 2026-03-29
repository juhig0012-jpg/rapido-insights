import pandas as pd

def load_data():
    bookings = pd.read_csv("data/bookings.csv")
    customers = pd.read_csv("data/customers.csv")
    drivers = pd.read_csv("data/drivers.csv")
    return bookings, customers, drivers

def clean_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Convert datetime
    df['booking_time'] = pd.to_datetime(df['booking_time'])

    return df