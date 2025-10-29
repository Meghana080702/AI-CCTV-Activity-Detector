import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load activity log
log_path = "alerts/activity_log.csv"

try:
    df = pd.read_csv(log_path)
except FileNotFoundError:
    print("No activity log found! Run ai_cctv_detector.py first to create it.")
    exit()

# Clean column names to remove spaces and hidden characters
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('\ufeff', '')  # removes BOM if present

# Now convert timestamp column safely
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
else:
    print("❌ 'Timestamp' column not found. Check your CSV headers:", df.columns)
    exit()


# Extract useful time info
df['Date'] = df['Timestamp'].dt.date
df['Hour'] = df['Timestamp'].dt.hour

# Show total detections over time
plt.figure(figsize=(10,5))
df['Date'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Total Detections per Day")
plt.xlabel("Date")
plt.ylabel("Number of Detections")
plt.show()

# Most common objects detected
plt.figure(figsize=(8,5))
df['Object'].value_counts().plot(kind='bar', color='salmon')
plt.title("Most Frequently Detected Objects")
plt.xlabel("Object")
plt.ylabel("Count")
plt.show()

# Detections by Hour
plt.figure(figsize=(10,5))
df['Hour'].value_counts().sort_index().plot(kind='line', marker='o', color='green')
plt.title("Activity Timeline by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Detections")
plt.grid(True)
plt.show()

print("\n✅ Dashboard loaded successfully!")
print("Total detections logged:", len(df))
