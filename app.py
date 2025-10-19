import pandas as pd
import streamlit as st
import plotly.express as px
from summary_client import GemmaSummaryClient, OpenAISummaryClient
import os

# --- Load Data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "kpi_data.csv")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"CSV not found at '{DATA_PATH}'.")
    st.stop()
df['date'] = pd.to_datetime(df['date'])

# --- Compute KPIs ---
latest = df.iloc[-1]
prev = df.iloc[-2]

revenue_growth = (latest['revenue'] - prev['revenue']) / prev['revenue'] * 100
user_growth = (latest['active_users'] - prev['active_users']) / prev['active_users'] * 100
churn_rate = latest['churned_users'] / latest['active_users'] * 100

# --- Display Dashboard ---
st.title("📈 Automated KPI Tracker")
st.write("Monitoring key business metrics with AI summaries")

col1, col2, col3 = st.columns(3)
col1.metric("Revenue", f"${latest['revenue']:,.0f}", f"{revenue_growth:.1f}% vs last week")
col2.metric("Active Users", f"{latest['active_users']:,}", f"{user_growth:.1f}%")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")

# --- Trend Charts ---
fig = px.line(df, x='date', y=['revenue', 'active_users'], markers=True)
st.plotly_chart(fig)

# --- AI Summary ---
st.subheader("🤖 Weekly Summary")

backend = st.selectbox("Choose AI model:", ["Gemma (Free, Local)", "OpenAI (API Key Required)"])

summary_text = f"""
Revenue growth: {revenue_growth:.2f}%
User growth: {user_growth:.2f}%
Churn rate: {churn_rate:.2f}%
"""

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