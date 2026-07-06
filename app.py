import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.title("K-Means Clustering on Iris Dataset")

# Load Iris dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Use only petal length and petal width
df = df[['petal length (cm)', 'petal width (cm)']]

st.subheader("Dataset")
st.write(df.head())

# Scatter plot before scaling
st.subheader("Original Data")

fig, ax = plt.subplots()
ax.scatter(df['petal length (cm)'], df['petal width (cm)'])
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
st.pyplot(fig)

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

st.subheader("Scaled Data")
st.write(scaled_df.head())

# Select K
k = st.slider("Select Number of Clusters (K)", 2, 10, 3)

# KMeans
km = KMeans(n_clusters=k, random_state=42)
scaled_df["Cluster"] = km.fit_predict(scaled_df)

# Cluster plot
st.subheader("K-Means Clusters")

fig, ax = plt.subplots()

colors = ['red', 'green', 'blue', 'orange', 'purple',
          'brown', 'pink', 'gray', 'cyan', 'olive']

for i in range(k):
    cluster = scaled_df[scaled_df["Cluster"] == i]
    ax.scatter(
        cluster['petal length (cm)'],
        cluster['petal width (cm)'],
        color=colors[i % len(colors)],
        label=f'Cluster {i}'
    )

# Centroids
centers = km.cluster_centers_
ax.scatter(
    centers[:, 0],
    centers[:, 1],
    color='black',
    marker='*',
    s=200,
    label='Centroids'
)

ax.set_xlabel("Scaled Petal Length")
ax.set_ylabel("Scaled Petal Width")
ax.legend()

st.pyplot(fig)

# Elbow Method
st.subheader("Elbow Plot")

sse = []

K = range(1, 11)

for i in K:
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(scaled_df[['petal length (cm)', 'petal width (cm)']])
    sse.append(model.inertia_)

fig, ax = plt.subplots()

ax.plot(K, sse, marker='o')
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("SSE (Inertia)")
ax.set_title("Elbow Method")

st.pyplot(fig)

st.success("The elbow plot usually suggests an optimal K around 3 for the Iris dataset.")