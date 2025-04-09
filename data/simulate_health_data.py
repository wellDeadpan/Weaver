import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# seed to ensure reproducibility
np.random.seed(42)

# total samples and anomalies
n_samples = 1000
n_anomalies = 50

# --------------------------
# Step 1: Simulate normal health data
# --------------------------
heart_rate = np.random.normal(loc=70, scale=5, size=n_samples)       # 正常心率
sleep_hours = np.random.normal(loc=7.5, scale=1.0, size=n_samples)   # 正常睡眠
step_count = np.random.normal(loc=8000, scale=2000, size=n_samples)  # 正常步数

# --------------------------
# Step 2: Simulate anomaly index
# --------------------------
anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

# High Heart Rate
heart_rate[anomaly_indices[:15]] += np.random.normal(loc=30, scale=5, size=15)

# Low sleep hours
sleep_hours[anomaly_indices[15:30]] -= np.random.normal(loc=3, scale=0.5, size=15)

# Low step counts
step_count[anomaly_indices[30:]] -= np.random.normal(loc=5000, scale=1000, size=20)

# --------------------------
# Step 3: Clip values to realistic ranges
# --------------------------
heart_rate = np.clip(heart_rate, 40, 180)
sleep_hours = np.clip(sleep_hours, 0, 12)
step_count = np.clip(step_count, 0, 20000)

# --------------------------
# Step 4: Build the DataFrame + Labeling
# --------------------------
df = pd.DataFrame({
    "heart_rate": heart_rate,
    "sleep_hours": sleep_hours,
    "step_count": step_count
})

# Normal by default
df["label"] = 0
# Label anomalies
df.loc[anomaly_indices, "label"] = 1
# label anomaly type
df["anomaly_type"] = 0
df.loc[anomaly_indices[:15], "anomaly_type"] = 1  # High Heart Rate
df.loc[anomaly_indices[15:30], "anomaly_type"] = 2  # Low Sleep Hours
df.loc[anomaly_indices[30:], "anomaly_type"] = 3  # Low Step Counts

# --------------------------
# Step 5: Save the DataFrame
# --------------------------
df.to_json("synthetic_health_anomaly_data.json", orient="records", indent=2)

print("✅ Data simulation complete. Saved to synthetic_health_anomaly_data.json")



plt.figure(figsize=(6, 4))
sns.countplot(x='anomaly_type', data=df)
plt.title("Anomaly Type Count")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='anomaly_type', y='heart_rate', data=df)
plt.title("Heart Rate by Anomaly Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df[df.label == 0], x='sleep_hours', label='Normal')
sns.kdeplot(data=df[df.label == 1], x='sleep_hours', label='Anomaly')
plt.title("Sleep Hours Distribution (Normal vs Anomaly)")
plt.legend()
plt.tight_layout()
plt.show()

cross_tab = pd.crosstab(df["label"], df["anomaly_type"])
sns.heatmap(cross_tab, annot=True, cmap="Blues")
plt.title("Label vs Anomaly Type")
plt.show()

