import pandas as pd
import streamlit as st
import plotly.express as px
from summary_client import GemmaSummaryClient, OpenAISummaryClient
import os

# --- Load Data ---
st.title("📊 Automated KPI Tracker")

# Load Google Analytics sample data
try:
    # Try to load the downloaded GA sample data first
    df = pd.read_csv("data/ga_sample_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    st.success("✅ Loaded Google Analytics sample data successfully")
except FileNotFoundError:
    st.info("GA sample data not found. Generating sample data...")
    # Generate sample data if file doesn't exist
    import numpy as np
    from datetime import datetime, timedelta
    
    start_date = datetime(2017, 7, 1)
    dates = [start_date + timedelta(days=i) for i in range(7)]
    
    np.random.seed(42)
    data = []
    base_sessions = 1000
    base_revenue = 10000
    
    for i, date in enumerate(dates):
        sessions = int(base_sessions + np.random.normal(0, 100))
        pageviews = int(sessions * (3 + np.random.normal(0, 0.5)))
        transactions = int(sessions * (0.05 + np.random.normal(0, 0.01)))
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
    st.success("✅ Generated Google Analytics sample data")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --- Compute KPIs ---
latest = df.iloc[-1]
prev = df.iloc[-2]

# Calculate growth rates for GA metrics
revenue_growth = (latest['revenue'] - prev['revenue']) / prev['revenue'] * 100 if prev['revenue'] > 0 else 0
sessions_growth = (latest['sessions'] - prev['sessions']) / prev['sessions'] * 100 if prev['sessions'] > 0 else 0
pageviews_growth = (latest['pageviews'] - prev['pageviews']) / prev['pageviews'] * 100 if prev['pageviews'] > 0 else 0

# Calculate conversion rate
conversion_rate = (latest['transactions'] / latest['sessions'] * 100) if latest['sessions'] > 0 else 0

# Calculate additional business metrics
total_revenue = df['revenue'].sum()
avg_daily_sessions = df['sessions'].mean()
best_day = df.loc[df['revenue'].idxmax()]
worst_day = df.loc[df['revenue'].idxmin()]
revenue_volatility = df['revenue'].std()
sessions_volatility = df['sessions'].std()

# Calculate week-over-week trends
if len(df) >= 2:
    w1_revenue = df.iloc[:len(df)//2]['revenue'].sum()
    w2_revenue = df.iloc[len(df)//2:]['revenue'].sum()
    weekly_revenue_growth = ((w2_revenue - w1_revenue) / w1_revenue * 100) if w1_revenue > 0 else 0
else:
    weekly_revenue_growth = 0

# --- Display Dashboard ---
st.title("📈 Automated KPI Tracker")
st.write("Monitoring key business metrics with AI summaries")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", f"${latest['revenue']:,.0f}", f"{revenue_growth:.1f}%")
col2.metric("Sessions", f"{latest['sessions']:,}", f"{sessions_growth:.1f}%")
col3.metric("Pageviews", f"{latest['pageviews']:,}", f"{pageviews_growth:.1f}%")
col4.metric("Conversion Rate", f"{conversion_rate:.1f}%")

# Additional metrics row
st.subheader("📊 Additional Insights")
col5, col6, col7, col8 = st.columns(4)
col5.metric("Total Revenue", f"${total_revenue:,.0f}")
col6.metric("Avg Daily Sessions", f"{avg_daily_sessions:,.0f}")
col7.metric("Revenue/Session", f"${(total_revenue / df['sessions'].sum()):.2f}")
col8.metric("Weekly Growth", f"{weekly_revenue_growth:.1f}%")

# --- Trend Charts ---
fig = px.line(df, x='date', y=['revenue', 'sessions', 'pageviews'], markers=True)
st.plotly_chart(fig)

# --- AI Summary ---
st.subheader("🤖 Weekly Summary")

backend = st.selectbox("Choose AI model:", ["Gemma (Free, Local)", "OpenAI (API Key Required)"])

summary_text = f"""
Please provide a comprehensive business analysis using the Google Analytics, Performance Summary and Key Insights below. Focuse on:
1. What are the key trends and patterns in this data?
2. What opportunities do you see for growth?
3. What should we be concerned about based on these metrics?
4. What specific, actionable recommendations should we implement?
5. How does the performance volatility affect our business strategy?

Google Analytics Performance Report:
Data Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}

Daily Performance:
{df[['date', 'revenue', 'sessions', 'pageviews', 'transactions']].to_string(index=False)}

Performance Summary:
- Total Revenue: ${total_revenue:,.0f}
- Average Daily Sessions: {avg_daily_sessions:,.0f}
- Best Performing Day: {best_day['date'].strftime('%Y-%m-%d')} (${best_day['revenue']:,.0f})
- Worst Performing Day: {worst_day['date'].strftime('%Y-%m-%d')} (${worst_day['revenue']:,.0f})
- Revenue Growth: {revenue_growth:.2f}%
- Sessions Growth: {sessions_growth:.2f}%
- Weekly Revenue Growth: {weekly_revenue_growth:.2f}%
- Revenue Volatility (std dev): ${revenue_volatility:,.0f}
- Sessions Volatility (std dev): {sessions_volatility:,.0f}

Key Insights:
- Conversion Rate: {conversion_rate:.1f}%
- Revenue per Session: ${(total_revenue / df['sessions'].sum()):.2f}
- Pageviews per Session: {(df['pageviews'].sum() / df['sessions'].sum()):.1f}
"""

# Debug: Show the prompt being generated
with st.expander("🔍 Debug: View Prompt Text (Copy to test in ChatGPT)", expanded=False):
    st.write("**Prompt that will be sent to AI:**")
    st.text_area("summary_text", summary_text, height=400, disabled=True, key="debug_prompt")
    st.write(f"**Prompt length:** {len(summary_text)} characters")
    st.write(f"**Data shape:** {df.shape}")
    st.write(f"**Data columns:** {', '.join(df.columns.tolist())}")
    
    # Show sample of the data
    st.write("**Sample data:**")
    st.dataframe(df.head())
    
    # Add copy button
    st.info("💡 Tip: Select all text above, copy it, and paste it into ChatGPT to test manually!")

if st.button("Generate Summary"):
    try:
        if "Gemma" in backend:
            with st.spinner("Loading model (first run may take several minutes)..."):
                client = GemmaSummaryClient()
            with st.spinner("Generating AI summary..."):
                summary = client.summarize(summary_text)
        else:
            with st.spinner("Generating AI summary..."):
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("Cannot use OpenAI: missing OPENAI_API_KEY.")
                    st.stop()
                client = OpenAISummaryClient(api_key=api_key)
                summary = client.summarize(summary_text)
        st.success(summary)
    except Exception as e:
        st.error(f"Failed to generate summary: {e}")