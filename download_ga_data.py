#!/usr/bin/env python3
"""
Download Google Analytics sample data from BigQuery and save as CSV
"""

import pandas as pd
from google.cloud import bigquery
import os

def download_ga_data():
    try:
        # Initialize BigQuery client
        client = bigquery.Client()
        
        # Query to get GA sample data
        query = """
        SELECT
            PARSE_DATE('%Y%m%d', _TABLE_SUFFIX) AS date,
            COUNT(DISTINCT visitId) AS sessions,
            SUM(totals.pageviews) AS pageviews,
            SUM(totals.transactions) AS transactions,
            SUM(totals.transactionRevenue)/1000000 AS revenue
        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170707'
        GROUP BY date
        ORDER BY date
        """
        
        print("Downloading Google Analytics sample data...")
        df = client.query(query).to_dataframe()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/ga_sample_data.csv', index=False)
        
        print(f"Successfully downloaded {len(df)} days of GA data!")
        print(f"Data saved to: data/ga_sample_data.csv")
        print("\nSample data:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nThis might be due to billing requirements.")
        print("Falling back to sample data generation...")
        return False

def create_sample_data():
    """Create sample GA-like data if BigQuery fails"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate 7 days of sample data
    start_date = datetime(2017, 7, 1)
    dates = [start_date + timedelta(days=i) for i in range(7)]
    
    # Generate realistic GA-like data
    np.random.seed(42)  # For reproducible results
    
    data = []
    base_sessions = 1000
    base_revenue = 10000
    
    for i, date in enumerate(dates):
        # Add some variation to make it realistic
        sessions = int(base_sessions + np.random.normal(0, 100))
        pageviews = int(sessions * (3 + np.random.normal(0, 0.5)))  # 3-4 pages per session
        transactions = int(sessions * (0.05 + np.random.normal(0, 0.01)))  # 5% conversion
        revenue = int(base_revenue + np.random.normal(0, 2000))
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'sessions': max(sessions, 0),
            'pageviews': max(pageviews, 0),
            'transactions': max(transactions, 0),
            'revenue': max(revenue, 0)
        })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/ga_sample_data.csv', index=False)
    
    print(f"Created sample GA data with {len(df)} days!")
    print(f"Data saved to: data/ga_sample_data.csv")
    print("\nSample data:")
    print(df.head())
    
    return True

if __name__ == "__main__":
    print("Downloading Google Analytics sample data...")
    
    # Try to download real data first
    if not download_ga_data():
        # Fall back to sample data
        create_sample_data()
