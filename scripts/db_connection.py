import os
import psycopg2
import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables from the .env file
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')


def connect_to_db():
    """Establish a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Connected to the database successfully")
        return connection
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        return None


def fetch_data(query):
    """Fetch data from PostgreSQL using a SQL query."""
    conn = connect_to_db()
    if conn:
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
        finally:
            conn.close()
    return None


def eda_nulls_outliers(df):
    """Perform basic EDA, check for nulls and outliers."""
    print("Checking for missing values...")
    print(df.isnull().sum())
    
    print("\nDescriptive statistics to check for outliers...")
    print(df.describe())

    return df


def clean_data(df):
    """Clean the dataset by replacing missing values and undefined values."""
    
    # Step 1: Replace 'undefined' with 'Unknown' for categorical columns
    categorical_columns = ['Handset Manufacturer', 'Handset Type', 'Last Location Name']
    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].replace('undefined', 'Unknown')
    
    # Step 2: Handle missing values in numeric columns
    for column in df.select_dtypes(include=[np.number]).columns:
        
        # Replace missing values (NaN) with the median
        median = df[column].median()
        df[column].fillna(median, inplace=True)
        
        # Calculate Z-scores to detect outliers
        z_scores = stats.zscore(df[column])
        
        # Define the threshold for outliers (e.g., Z-score > 3 or Z-score < -3)
        threshold = 3
        
        # Replace outliers with the median
        df[column] = np.where(np.abs(z_scores) > threshold, median, df[column])
    
    # Step 3: Handle missing and invalid 'Start' and 'End' dates
    # Convert 'Start' and 'End' columns to datetime format
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')  # Convert invalid dates to NaT
    df['End'] = pd.to_datetime(df['End'], errors='coerce')      # Convert invalid dates to NaT
    
    # Fill missing dates with a placeholder or median date
    # Option 1: Fill missing dates with a specific placeholder (e.g., 'Unknown')
    df['Start'].fillna(pd.Timestamp('1970-01-01 00:00:00'), inplace=True)  # Example placeholder
    df['End'].fillna(pd.Timestamp('1970-01-01 00:00:00'), inplace=True)    # Example placeholder
    
    # Option 2: Alternatively, you can drop rows with missing 'Start' or 'End' dates
    # df.dropna(subset=['Start', 'End'], inplace=True)
    
    # Step 4: Replace any remaining null values in categorical columns with 'Unknown'
    for column in categorical_columns:
        if column in df.columns:
            df[column].fillna('Unknown', inplace=True)
    
    return df



# def convert_bytes_to_megabytes(df):
#     """Convert relevant columns from bytes to megabytes."""
#     # List of columns to convert
#     byte_columns = [
#         'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
#         'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
#         'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)',
#         'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
#         'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',
#         'Total UL (Bytes)', 'Total DL (Bytes)'
#     ]
    
#     # Conversion factor: 1 MB = 1024^2 bytes
#     bytes_to_mb = 1024 ** 2
    
#     # Convert each column from bytes to megabytes
#     for column in byte_columns:
#         if column in df.columns:
#             df[column] = df[column] / bytes_to_mb
    
#     return df


def aggregate_user_behavior(df):
    """
    Aggregate user behavior per user.
    This function will return a DataFrame with aggregated data for each user including:
    - Number of xDR sessions
    - Total session duration
    - Total download (DL) and upload (UL) data
    - Total data volume (in Bytes) for each application
    """
    
    # Aggregate per user
    aggregated_data = df.groupby('MSISDN/Number').agg(
        number_of_xdr_sessions=('Bearer Id', 'count'),  # Count the number of xDR sessions
        total_session_duration=('Dur. (ms)', 'sum'),  # Total session duration
        total_download_data=('Total DL (Bytes)', 'sum'),  # Total download data
        total_upload_data=('Total UL (Bytes)', 'sum'),  # Total upload data
        
        # Total data volume for each application
        social_media_dl=('Social Media DL (Bytes)', 'sum'),
        google_dl=('Google DL (Bytes)', 'sum'),
        email_dl=('Email DL (Bytes)', 'sum'),
        youtube_dl=('Youtube DL (Bytes)', 'sum'),
        netflix_dl=('Netflix DL (Bytes)', 'sum'),
        gaming_dl=('Gaming DL (Bytes)', 'sum'),
        other_dl=('Other DL (Bytes)', 'sum'),
        social_media_ul=('Social Media UL (Bytes)', 'sum'),
        google_ul=('Google UL (Bytes)', 'sum'),
        email_ul=('Email UL (Bytes)', 'sum'),
        youtube_ul=('Youtube UL (Bytes)', 'sum'),
        netflix_ul=('Netflix UL (Bytes)', 'sum'),
        gaming_ul=('Gaming UL (Bytes)', 'sum'),
        other_ul=('Other UL (Bytes)', 'sum')
    ).reset_index()

    return aggregated_data
