import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("K-Means Clustering using Income Dataset")

# Load dataset
df = pd.read_csv("income.csv")

st.subheader("Original Dataset")
st.dataframe(df)

# -------------------------------
# Scatter Plot Before Clustering
# -------------------------------
st.subheader("Scatter Plot (Original Data)")

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(df["Age"], df["Income($)"], color="blue")
ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
st.pyplot(fig)

# -------------------------------
# KMeans Before Scaling
# -------------------------------
st.subheader("K-Means Clustering (Before Scaling)")

km = KMeans(n_clusters=3, random_state=42)

df_before = df.copy()
df_before["Cluster"] = km.fit_predict(df_before[["Age", "Income($)"]])

centers_before = km.cluster_centers_

fig, ax = plt.subplots(figsize=(6,5))

colors = ["green", "red", "black"]

for i in range(3):
    cluster = df_before[df_before["Cluster"] == i]
    ax.scatter(
        cluster["Age"],
        cluster["Income($)"],
        color=colors[i],
        label=f"Cluster {i}"
    )

ax.scatter(
    centers_before[:,0],
    centers_before[:,1],
    color="purple",
    marker="*",
    s=250,
    label="Centroids"
)

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
ax.legend()

st.pyplot(fig)

st.write("Cluster Centers")
st.write(pd.DataFrame(centers_before, columns=["Age", "Income"]))

# -------------------------------
# Scaling
# -------------------------------
st.subheader("Min-Max Scaling")

scaled_df = df.copy()

age_scaler = MinMaxScaler()
income_scaler = MinMaxScaler()

scaled_df["Age"] = age_scaler.fit_transform(scaled_df[["Age"]])
scaled_df["Income($)"] = income_scaler.fit_transform(scaled_df[["Income($)"]])

st.write(scaled_df)

# -------------------------------
# Scatter After Scaling
# -------------------------------
st.subheader("Scatter Plot After Scaling")

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(scaled_df["Age"], scaled_df["Income($)"], color="blue")
ax.set_xlabel("Scaled Age")
ax.set_ylabel("Scaled Income")
st.pyplot(fig)

# -------------------------------
# KMeans After Scaling
# -------------------------------
st.subheader("K-Means Clustering (After Scaling)")

km = KMeans(n_clusters=3, random_state=42)

scaled_df["Cluster"] = km.fit_predict(
    scaled_df[["Age", "Income($)"]]
)

centers_after = km.cluster_centers_

fig, ax = plt.subplots(figsize=(6,5))

for i in range(3):
    cluster = scaled_df[scaled_df["Cluster"] == i]
    ax.scatter(
        cluster["Age"],
        cluster["Income($)"],
        color=colors[i],
        label=f"Cluster {i}"
    )

ax.scatter(
    centers_after[:,0],
    centers_after[:,1],
    color="purple",
    marker="*",
    s=250,
    label="Centroids"
)

ax.set_xlabel("Scaled Age")
ax.set_ylabel("Scaled Income")
ax.legend()

st.pyplot(fig)

st.write("Cluster Centers After Scaling")
st.write(pd.DataFrame(centers_after, columns=["Age", "Income"]))

# -------------------------------
# Elbow Method
# -------------------------------
st.subheader("Elbow Method")

sse = []

k_range = range(1, 10)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_df[["Age", "Income($)"]])
    sse.append(model.inertia_)

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(k_range, sse, marker="o")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Sum of Squared Error (SSE)")
ax.set_title("Elbow Plot")

st.pyplot(fig)

st.success("The elbow point suggests the optimal number of clusters.")