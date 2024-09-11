import psycopg2
import pandas as pd
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv


# Load environment variables for database connection (or hardcode them)
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
load_dotenv()
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
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def fetch_data_from_db(query):
    """Fetch data from PostgreSQL using a SQL query."""
    conn = connect_to_db()
    if conn:
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
        finally:
            conn.close()
    return None
# Define SQL queries to fetch data for each task
def get_user_overview_data():
    query = "SELECT * FROM xdr_data"  
    return fetch_data_from_db(query)

def get_user_engagement_data():
    query = "SELECT * FROM user_scores"  
    return fetch_data_from_db(query)

def get_experience_analysis_data():
    query = "SELECT * FROM user_scores"  
    return fetch_data_from_db(query)

def get_satisfaction_analysis_data():
    query = "SELECT * FROM user_scores"  
    return fetch_data_from_db(query)

# Streamlit App Layout
st.title("Telecommunication User Analysis Dashboard")

# Navigation Bar
option = st.sidebar.selectbox(
    "Select a Page",
    ("User Overview Analysis", "User Engagement Analysis", "Experience Analysis", "Satisfaction Analysis")
)

# Page: User Overview Analysis
if option == "User Overview Analysis":
    st.header("User Overview Analysis")
    user_overview_df = get_user_overview_data()
    
    if user_overview_df is not None:
        st.write(user_overview_df.head())  # Display the first few rows of data
        
        # Plot the top 10 handsets by usage
        fig, ax = plt.subplots()
        sns.countplot(y='Handset Type', data=user_overview_df, order=user_overview_df['Handset Type'].value_counts().iloc[:10].index, ax=ax)
        ax.set_title("Top 10 Handsets Used by Customers")
        st.pyplot(fig)
    else:
        st.warning("No data found for User Overview Analysis.")

# Page: User Engagement Analysis
elif option == "User Engagement Analysis":
    st.header("User Engagement Analysis")
    user_engagement_df = get_user_engagement_data()
    
    if user_engagement_df is not None:
        st.write(user_engagement_df.head())
        
        # Plot user engagement metrics
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability
        sns.barplot(x='MSISDN/Number', y='session_frequency', data=user_engagement_df.head(10), ax=ax)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)  # Rotate labels 90 degrees
        
        ax.set_title("Top 10 Customers by Session Frequency")
        ax.set_xlabel("MSISDN/Number")
        ax.set_ylabel("Session Frequency")
        
        # Optionally, you can reduce the number of x-ticks shown
        # ax.set_xticks(ax.get_xticks()[::2])  # Show every second tick, optional if needed
        
        st.pyplot(fig)
        
    else:
        st.warning("No data found for User Engagement Analysis.")

# Page: Experience Analysis
elif option == "Experience Analysis":
    st.header("Experience Analysis")
    experience_analysis_df = get_experience_analysis_data()
    
    if experience_analysis_df is not None:
        st.write(experience_analysis_df.head())
        
        # Get the top 10 most frequent handset types
        top_handsets = experience_analysis_df['Handset_Type'].value_counts().nlargest(10).index
        
        # Filter the DataFrame to only include these handset types
        filtered_data = experience_analysis_df[experience_analysis_df['Handset_Type'].isin(top_handsets)]
        
        # Plot distribution of average throughput by handset type
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size if needed
        sns.boxplot(x='Handset_Type', y='avg_throughput', data=filtered_data, ax=ax)
        ax.set_title("Distribution of Average Throughput by Top 10 Handset Types")
        plt.xticks(rotation=90)  # Rotate x-axis labels
        st.pyplot(fig)
    else:
        st.warning("No data found for Experience Analysis.")




# Page: Satisfaction Analysis
elif option == "Satisfaction Analysis":
    st.header("Satisfaction Analysis")
    satisfaction_analysis_df = get_satisfaction_analysis_data()
    
    if satisfaction_analysis_df is not None:
        st.write(satisfaction_analysis_df.head())
        
        # Plot Satisfaction Score per User
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability
        sns.barplot(x='MSISDN/Number', y='satisfaction_score', data=satisfaction_analysis_df.head(10), ax=ax)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)  # Rotate labels 90 degrees
        
        ax.set_title("Top 10 Users by Satisfaction Score")
        ax.set_xlabel("MSISDN/Number")
        ax.set_ylabel("Satisfaction Score")
        
        # Optionally, reduce the number of x-ticks shown
        # ax.set_xticks(ax.get_xticks()[::2])  # Show every second tick, optional if needed
        
        st.pyplot(fig)
    else:
        st.warning("No data found for Satisfaction Analysis.")
