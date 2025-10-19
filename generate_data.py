import pandas as pd
from datetime import datetime, timedelta
import random

# Generate 5 weeks of fake KPI data
start_date = datetime(2025, 9, 1)
data = []
revenue = 10000
active_users = 1200
signups = 300
churn = 50

for i in range(5):
    date = start_date + timedelta(weeks=i)
    revenue += random.randint(500, 2000)
    active_users += random.randint(-50, 100)
    signups = random.randint(250, 450)
    churn = random.randint(40, 60)
    data.append([date.date(), revenue, active_users, signups, churn])

df = pd.DataFrame(data, columns=["date", "revenue", "active_users", "new_signups", "churned_users"])
df.to_csv("kpi_data.csv", index=False)
print("✅ kpi_data.csv created successfully!")