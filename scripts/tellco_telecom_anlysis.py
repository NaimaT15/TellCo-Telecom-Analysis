import os
import warnings
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load environment variables from the .env file
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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


def calculate_user_engagement(df):
    """
    This function calculates user engagement metrics such as:
    - Session frequency (number of xDR sessions)
    - Total session duration
    - Total traffic (download + upload data)
    """
    
    # Ensure 'Total DL (Bytes)' and 'Total UL (Bytes)' are numeric, and handle non-numeric values
    df['Total DL (Bytes)'] = pd.to_numeric(df['Total DL (Bytes)'], errors='coerce')
    df['Total UL (Bytes)'] = pd.to_numeric(df['Total UL (Bytes)'], errors='coerce')
    
    # Fill NaN values with 0
    df['Total DL (Bytes)'].fillna(0, inplace=True)
    df['Total UL (Bytes)'].fillna(0, inplace=True)

    # Aggregate user engagement metrics (calculate total DL and UL separately)
    user_engagement = df.groupby('MSISDN/Number').agg(
        session_frequency=('Bearer Id', 'count'),  # Count the number of sessions
        total_session_duration=('Dur. (ms)', 'sum'),  # Sum of session duration
        total_download_data=('Total DL (Bytes)', 'sum'),  # Sum of download data
        total_upload_data=('Total UL (Bytes)', 'sum')  # Sum of upload data
    ).reset_index()

    # Calculate total traffic by adding download and upload traffic
    user_engagement['total_traffic'] = user_engagement['total_download_data'] + user_engagement['total_upload_data']

    return user_engagement

def calculate_application_traffic(df):
    """
    This function calculates the total traffic contributed by each application (Social Media, Google, YouTube, etc.)
    per user and returns a DataFrame with total traffic and contribution percentages.
    """

    # Ensure relevant application data columns are numeric
    app_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)',
        'Gaming DL (Bytes)', 'Other DL (Bytes)', 'Social Media UL (Bytes)', 'Google UL (Bytes)',
        'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 'Other UL (Bytes)'
    ]
    for col in app_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate total traffic per application (Download + Upload) for each user
    df['Social Media Traffic'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
    df['Google Traffic'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)']
    df['YouTube Traffic'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
    df['Netflix Traffic'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
    df['Gaming Traffic'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']
    df['Other Traffic'] = df['Other DL (Bytes)'] + df['Other UL (Bytes)']

    # Calculate total traffic for each user
    df['Total Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

    # Aggregate data per user
    user_application_traffic = df.groupby('MSISDN/Number').agg(
        total_traffic=('Total Traffic', 'sum'),
        social_media_traffic=('Social Media Traffic', 'sum'),
        google_traffic=('Google Traffic', 'sum'),
        youtube_traffic=('YouTube Traffic', 'sum'),
        netflix_traffic=('Netflix Traffic', 'sum'),
        gaming_traffic=('Gaming Traffic', 'sum'),
        other_traffic=('Other Traffic', 'sum')
    ).reset_index()

    # Calculate percentage contribution of each application to the total traffic
    user_application_traffic['Social Media %'] = (user_application_traffic['social_media_traffic'] / user_application_traffic['total_traffic']) * 100
    user_application_traffic['Google %'] = (user_application_traffic['google_traffic'] / user_application_traffic['total_traffic']) * 100
    user_application_traffic['YouTube %'] = (user_application_traffic['youtube_traffic'] / user_application_traffic['total_traffic']) * 100
    user_application_traffic['Netflix %'] = (user_application_traffic['netflix_traffic'] / user_application_traffic['total_traffic']) * 100
    user_application_traffic['Gaming %'] = (user_application_traffic['gaming_traffic'] / user_application_traffic['total_traffic']) * 100
    user_application_traffic['Other %'] = (user_application_traffic['other_traffic'] / user_application_traffic['total_traffic']) * 100

    return user_application_traffic

def calculate_engagement_over_time(df):
    """
    This function calculates user engagement trends over time based on session start and end times.
    The result will show total traffic and session duration per day.
    """
    
    # Convert 'Start' and 'End' columns to datetime format
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['End'] = pd.to_datetime(df['End'], errors='coerce')
    
    # Ensure 'Total DL (Bytes)' and 'Total UL (Bytes)' are numeric
    df['Total DL (Bytes)'] = pd.to_numeric(df['Total DL (Bytes)'], errors='coerce').fillna(0)
    df['Total UL (Bytes)'] = pd.to_numeric(df['Total UL (Bytes)'], errors='coerce').fillna(0)
    
    # Extract the date part (without time) for grouping
    df['Start_Date'] = df['Start'].dt.date
    df['End_Date'] = df['End'].dt.date

    # Calculate total traffic per date (Download + Upload)
    df['Total Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

    # Aggregate traffic and session duration by start date
    engagement_over_time = df.groupby('Start_Date').agg(
        total_traffic=('Total Traffic', 'sum'),
        total_session_duration=('Dur. (ms)', 'sum')
    ).reset_index()

    return engagement_over_time


def aggregate_metrics_per_customer(df):
    """
    This function aggregates session frequency, total session duration, and total traffic per customer (MSISDN).
    """
    # Aggregate metrics
    customer_metrics = df.groupby('MSISDN/Number').agg(
        session_frequency=('Bearer Id', 'count'),  # Count number of sessions
        total_session_duration=('Dur. (ms)', 'sum'),  # Total session duration
        total_download=('Total DL (Bytes)', 'sum'),  # Total download data
        total_upload=('Total UL (Bytes)', 'sum')  # Total upload data
    ).reset_index()

    # Calculate total traffic as download + upload
    customer_metrics['total_traffic'] = customer_metrics['total_download'] + customer_metrics['total_upload']

    return customer_metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def normalize_and_cluster(df):
    """
    Normalize engagement metrics and run K-Means clustering with k=3.
    """
    # Select metrics for normalization
    metrics_to_normalize = df[['session_frequency', 'total_session_duration', 'total_traffic']]

    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(metrics_to_normalize)

    # Run K-Means with k=3
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['cluster'] = kmeans.fit_predict(normalized_metrics)

    return df, kmeans

def cluster_statistics(df):
    """
    Compute minimum, maximum, average, and total values for each metric per cluster.
    """
    cluster_stats = df.groupby('cluster').agg(
        min_session_frequency=('session_frequency', 'min'),
        max_session_frequency=('session_frequency', 'max'),
        avg_session_frequency=('session_frequency', 'mean'),
        total_session_frequency=('session_frequency', 'sum'),

        min_session_duration=('total_session_duration', 'min'),
        max_session_duration=('total_session_duration', 'max'),
        avg_session_duration=('total_session_duration', 'mean'),
        total_session_duration=('total_session_duration', 'sum'),

        min_total_traffic=('total_traffic', 'min'),
        max_total_traffic=('total_traffic', 'max'),
        avg_total_traffic=('total_traffic', 'mean'),
        total_total_traffic=('total_traffic', 'sum')
    ).reset_index()

    return cluster_stats


def aggregate_application_traffic(df):
    """
    Aggregate total traffic per application for each customer and report the top 10 most engaged users per application.
    """
    # Calculate total traffic for each application
    df['social_media_traffic'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
    df['google_traffic'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)']
    df['youtube_traffic'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
    df['netflix_traffic'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
    df['gaming_traffic'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']

    # Aggregate traffic per user
    app_traffic = df.groupby('MSISDN/Number').agg(
        total_social_media=('social_media_traffic', 'sum'),
        total_google=('google_traffic', 'sum'),
        total_youtube=('youtube_traffic', 'sum'),
        total_netflix=('netflix_traffic', 'sum'),
        total_gaming=('gaming_traffic', 'sum')
    ).reset_index()

    return app_traffic


def elbow_method(df):
    """
    Apply the elbow method to determine the optimal value of k for K-Means clustering.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt

    # Normalize the data
    metrics_to_normalize = df[['session_frequency', 'total_session_duration', 'total_traffic']]
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(metrics_to_normalize)

    # Run K-Means for different values of k
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(normalized_metrics)
        inertia.append(kmeans.inertia_)

    # Plot the inertia values for the elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()


def top_bottom_frequent_values(df, column, n=10):
    """
    This function computes and returns the top, bottom, and most frequent values for a specified column.

    :param df: DataFrame containing the data
    :param column: The column name for which to compute the top, bottom, and frequent values
    :param n: The number of values to return (default is 10)
    :return: A dictionary containing top, bottom, and most frequent values
    """
    top_values = df[column].nlargest(n)
    bottom_values = df[column].nsmallest(n)
    most_frequent_values = df[column].value_counts().head(n)
    
    return {
        'top_values': top_values,
        'bottom_values': bottom_values,
        'most_frequent_values': most_frequent_values
    }

def experience_analytics(df):
    """
    Perform user experience analysis based on network parameters.

    :param df: DataFrame containing telecommunication data
    :return: Aggregated DataFrame with average TCP retransmission, RTT, throughput, and handset type per customer.
    """

    # Treat missing values: Replace missing with mean (for numeric columns) and mode (for categorical columns)
    df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
    df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean(), inplace=True)
    df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
    df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean(), inplace=True)
    df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
    df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
    df['Avg Bearer TP UL (kbps)'].fillna(df['Avg Bearer TP UL (kbps)'].mean(), inplace=True)

    # Aggregate metrics per customer
    experience_df = df.groupby('MSISDN/Number').agg(
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean'),
        avg_rtt=('Avg RTT DL (ms)', 'mean'),
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean'),
        handset_type=('Handset Type', lambda x: x.mode()[0])  # Mode is used to get the most common handset per customer
    ).reset_index()

    return experience_df

def display_top_bottom_frequent(df):
    """
    This function computes and displays the top, bottom, and most frequent values for TCP, RTT, and Throughput.
    
    :param df: DataFrame containing telecommunication data
    """

    # Top, bottom, and most frequent TCP values
    tcp_values = top_bottom_frequent_values(df, 'TCP DL Retrans. Vol (Bytes)')
    print("TCP Retransmission Values:")
    print(f"Top 10:\n{tcp_values['top_values']}")
    print(f"Bottom 10:\n{tcp_values['bottom_values']}")
    print(f"Most Frequent:\n{tcp_values['most_frequent_values']}\n")

    # Top, bottom, and most frequent RTT values
    rtt_values = top_bottom_frequent_values(df, 'Avg RTT DL (ms)')
    print("RTT Values:")
    print(f"Top 10:\n{rtt_values['top_values']}")
    print(f"Bottom 10:\n{rtt_values['bottom_values']}")
    print(f"Most Frequent:\n{rtt_values['most_frequent_values']}\n")

    # Top, bottom, and most frequent Throughput values
    throughput_values = top_bottom_frequent_values(df, 'Avg Bearer TP DL (kbps)')
    print("Throughput Values:")
    print(f"Top 10:\n{throughput_values['top_values']}")
    print(f"Bottom 10:\n{throughput_values['bottom_values']}")
    print(f"Most Frequent:\n{throughput_values['most_frequent_values']}\n")


def throughput_tcp_per_handset(df):
    """
    Compute the distribution of average throughput and TCP retransmission per handset type.

    :param df: DataFrame containing telecommunication data
    :return: DataFrames with average throughput and TCP retransmission per handset type
    """
    # Calculate average throughput per handset type
    throughput_per_handset = df.groupby('Handset Type').agg(
        avg_throughput=('Avg Bearer TP DL (kbps)', 'mean')
    ).reset_index()

    # Calculate average TCP retransmission per handset type
    tcp_per_handset = df.groupby('Handset Type').agg(
        avg_tcp_retransmission=('TCP DL Retrans. Vol (Bytes)', 'mean')
    ).reset_index()

    return throughput_per_handset, tcp_per_handset

def plot_distributions(throughput_df, tcp_df):
    """
    Plot the distribution of average throughput and TCP retransmission per handset type.

    :param throughput_df: DataFrame containing average throughput per handset type
    :param tcp_df: DataFrame containing average TCP retransmission per handset type
    """

    # Plot throughput distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(throughput_df['avg_throughput'], bins=30, kde=True)
    plt.title('Distribution of Average Throughput per Handset Type')
    plt.xlabel('Average Throughput (kbps)')
    plt.ylabel('Frequency')
    plt.show()

    # Plot TCP retransmission distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(tcp_df['avg_tcp_retransmission'], bins=30, kde=True)
    plt.title('Distribution of Average TCP Retransmission per Handset Type')
    plt.xlabel('Average TCP Retransmission (Bytes)')
    plt.ylabel('Frequency')
    plt.show()

def report_throughput_tcp(throughput_df, tcp_df):
    """
    Print the interpretation of the findings for average throughput and TCP retransmission per handset type.

    :param throughput_df: DataFrame containing average throughput per handset type
    :param tcp_df: DataFrame containing average TCP retransmission per handset type
    """
    # Interpret the throughput findings
    print("Interpretation of Average Throughput per Handset Type:")
    top_throughput_handset = throughput_df.nlargest(1, 'avg_throughput')
    low_throughput_handset = throughput_df.nsmallest(1, 'avg_throughput')
    print(f"Handset Type with the highest average throughput: {top_throughput_handset.iloc[0]['Handset Type']} ({top_throughput_handset.iloc[0]['avg_throughput']} kbps)")
    print(f"Handset Type with the lowest average throughput: {low_throughput_handset.iloc[0]['Handset Type']} ({low_throughput_handset.iloc[0]['avg_throughput']} kbps)\n")

    # Interpret the TCP retransmission findings
    print("Interpretation of Average TCP Retransmission per Handset Type:")
    top_tcp_handset = tcp_df.nlargest(1, 'avg_tcp_retransmission')
    low_tcp_handset = tcp_df.nsmallest(1, 'avg_tcp_retransmission')
    print(f"Handset Type with the highest average TCP retransmission: {top_tcp_handset.iloc[0]['Handset Type']} ({top_tcp_handset.iloc[0]['avg_tcp_retransmission']} Bytes)")
    print(f"Handset Type with the lowest average TCP retransmission: {low_tcp_handset.iloc[0]['Handset Type']} ({low_tcp_handset.iloc[0]['avg_tcp_retransmission']} Bytes)")


def preprocess_data_for_clustering(df):
    """
    Preprocess the data for k-means clustering: 
    - Normalize numeric experience metrics (TCP retransmission, RTT, throughput)
    - One-hot encode the handset type
    """
    # Select relevant experience metrics
    experience_metrics = df[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput', 'handset_type']].copy()
    
    # Normalize the numeric columns (TCP, RTT, throughput)
    scaler = StandardScaler()
    experience_metrics.loc[:, ['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']] = scaler.fit_transform(
        experience_metrics[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']]
    )
    
    # One-hot encode the handset type
    encoder = OneHotEncoder()
    handset_encoded = pd.DataFrame(encoder.fit_transform(experience_metrics[['handset_type']]).toarray(), index=experience_metrics.index)
    
    # Merge the encoded handset data with the normalized metrics
    experience_metrics = experience_metrics.drop('handset_type', axis=1)
    experience_metrics = pd.concat([experience_metrics, handset_encoded], axis=1)
    
    # Ensure all column names are strings
    experience_metrics.columns = experience_metrics.columns.astype(str)
    
    return experience_metrics



def perform_kmeans_clustering(df, k=3):
    """
    Perform K-Means clustering on the preprocessed data.
    
    :param df: Preprocessed DataFrame containing normalized experience metrics and one-hot encoded handset type
    :param k: Number of clusters for K-Means (default is 3)
    :return: Cluster labels for each user
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(df)
    
    return clusters

def describe_clusters(df, clusters):
    """
    Provide a brief description of each cluster based on the experience metrics.
    
    :param df: Original DataFrame containing experience metrics
    :param clusters: Cluster labels for each user
    """
    df['cluster'] = clusters
    
    # Group by cluster and compute descriptive statistics for each cluster
    cluster_summary = df.groupby('cluster').agg(
        avg_tcp_retransmission=('avg_tcp_retransmission', 'mean'),
        avg_rtt=('avg_rtt', 'mean'),
        avg_throughput=('avg_throughput', 'mean'),
        handset_type=('handset_type', lambda x: x.mode()[0])  # Most frequent handset type
    ).reset_index()
    
    # Display the cluster descriptions
    for idx, row in cluster_summary.iterrows():
        print(f"Cluster {row['cluster']} Description:")
        print(f"  - Average TCP Retransmission: {row['avg_tcp_retransmission']:.2f}")
        print(f"  - Average RTT: {row['avg_rtt']:.2f} ms")
        print(f"  - Average Throughput: {row['avg_throughput']:.2f} kbps")
        print(f"  - Most Common Handset Type: {row['handset_type']}")
        print("\n")




def assign_scores(user_data, engagement_centroids, experience_centroids):
    """
    Assign engagement and experience scores to users based on Euclidean distance from the least engaged
    and worst experience clusters.
    
    Parameters:
    - user_data: DataFrame containing the user metrics (e.g., engagement, experience metrics).
    - engagement_centroids: Centroids of engagement clusters.
    - experience_centroids: Centroids of experience clusters.
    
    Returns:
    - DataFrame with engagement and experience scores.
    """
    
    # Identify the index of the least engaged and worst experience clusters
    least_engaged_cluster = np.argmin(np.sum(engagement_centroids, axis=1))
    worst_experience_cluster = np.argmax(np.sum(experience_centroids, axis=1))
    
    # Initialize lists to store engagement and experience scores
    engagement_scores = []
    experience_scores = []
    
    # Iterate over each user and calculate the Euclidean distances
    for i, row in user_data.iterrows():
        # Extract the user's engagement and experience data
        user_engagement_data = row[['session_frequency', 'total_session_duration', 'total_traffic']]
        user_experience_data = row[['avg_tcp_retransmission', 'avg_rtt', 'avg_throughput']]
        
        # Calculate the engagement score (distance from least engaged cluster)
        engagement_score = np.linalg.norm(user_engagement_data.values - engagement_centroids[least_engaged_cluster])
        engagement_scores.append(engagement_score)
        
        # Calculate the experience score (distance from worst experience cluster)
        experience_score = np.linalg.norm(user_experience_data.values - experience_centroids[worst_experience_cluster])
        experience_scores.append(experience_score)
    
    # Assign the scores to the user data
    user_data['engagement_score'] = engagement_scores
    user_data['experience_score'] = experience_scores
    
    return user_data


def calculate_satisfaction_score(user_data):
    """
    Calculate the satisfaction score as the average of the engagement and experience scores,
    and return the top 10 satisfied users.
    
    Parameters:
    - user_data: DataFrame containing user engagement and experience scores.
    
    Returns:
    - DataFrame with the top 10 satisfied customers.
    """
    # Calculate the satisfaction score as the average of engagement and experience scores
    user_data['satisfaction_score'] = (user_data['engagement_score'] + user_data['experience_score']) / 2
    
    # Sort the users by satisfaction score in descending order
    top_10_satisfied_customers = user_data[['MSISDN/Number', 'satisfaction_score']].sort_values(
        by='satisfaction_score', ascending=False).head(10)
        
    return top_10_satisfied_customers
def run_kmeans_on_scores(user_scores, k=2):

    # Extract the engagement and experience scores
    score_data = user_scores[['engagement_score', 'experience_score']]
    
    # Standardize the scores
    scaler = StandardScaler()
    score_data_scaled = scaler.fit_transform(score_data)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    user_scores['cluster'] = kmeans.fit_predict(score_data_scaled)
    
    # Return the updated DataFrame with cluster labels
    return user_scores, kmeans.cluster_centers_
def aggregate_scores_per_cluster(user_scores):

    # Group by the 'cluster' column and compute the average satisfaction and experience scores
    cluster_aggregates = user_scores.groupby('cluster').agg(
        avg_satisfaction_score=('satisfaction_score', 'mean'),
        avg_experience_score=('experience_score', 'mean')
    ).reset_index()
    
    return cluster_aggregates


def create_postgres_engine():
    """Create an SQLAlchemy engine for PostgreSQL."""
    try:
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        print("Connected to the PostgreSQL database successfully")
        return engine
    except Exception as e:
        print(f"Error connecting to the PostgreSQL database: {e}")
        return None

# Function to export the DataFrame to PostgreSQL
def export_to_postgres(user_scores):
    """Export the DataFrame to the PostgreSQL database."""
    engine = create_postgres_engine()
    if engine:
        try:
            # Export the DataFrame to the PostgreSQL table 'user_scores'
            user_scores.to_sql('user_scores', con=engine, if_exists='replace', index=False)
            print("Data exported successfully to PostgreSQL database.")
        except Exception as e:
            print(f"Error exporting data: {e}")
def fetch_exported_data():
    """Fetch data from the PostgreSQL database to verify export."""
    engine = create_postgres_engine()
    if engine:
        try:
            query = "SELECT * FROM user_scores LIMIT 10;"
            df = pd.read_sql(query, engine)
            print(df)
        except Exception as e:
            print(f"Error fetching exported data: {e}")


