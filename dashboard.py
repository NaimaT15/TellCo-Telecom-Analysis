import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (assuming it's saved as 'cleaned_data.csv')
df = pd.read_csv('cleaned_data.csv')

# Title for your dashboard
st.title('Telecommunication Analysis Dashboard')

# Sidebar for navigation
st.sidebar.title('Navigation')
option = st.sidebar.selectbox('Select Page:', ['User Overview Analysis', 'User Engagement Analysis', 'Experience Analysis', 'Satisfaction Analysis'])

# Function to show each page based on selection
def show_user_overview():
    st.header('User Overview Analysis')
    st.write('This section contains insights from the user overview analysis.')
    
    # Add KPIs or any key metrics here for the user overview
    top_handsets = df['Handset Type'].value_counts().head(10)
    st.write('Top 10 Handsets Used by Customers:', top_handsets)
    
    # Plotting
    fig, ax = plt.subplots()
    top_handsets.plot(kind='bar', ax=ax)
    st.pyplot(fig)

def show_user_engagement():
    st.header('User Engagement Analysis')
    st.write('This section contains insights on user engagement analysis.')
    
    # Example plot: Distribution of Session Frequency
    fig, ax = plt.subplots()
    sns.histplot(df['session_frequency'], kde=True, ax=ax)
    st.pyplot(fig)
    
def show_experience_analysis():
    st.header('Experience Analysis')
    st.write('This section contains insights on user experience analysis.')
    
    # Example plot: TCP Retransmission Distribution
    fig, ax = plt.subplots()
    sns.histplot(df['avg_tcp_retransmission'], kde=True, ax=ax)
    st.pyplot(fig)
    
def show_satisfaction_analysis():
    st.header('Satisfaction Analysis')
    st.write('This section contains insights from the satisfaction analysis.')
    
    # Example: Satisfaction score distribution
    fig, ax = plt.subplots()
    sns.histplot(df['satisfaction_score'], kde=True, ax=ax)
    st.pyplot(fig)

# Display pages based on user selection
if option == 'User Overview Analysis':
    show_user_overview()
elif option == 'User Engagement Analysis':
    show_user_engagement()
elif option == 'Experience Analysis':
    show_experience_analysis()
elif option == 'Satisfaction Analysis':
    show_satisfaction_analysis()
