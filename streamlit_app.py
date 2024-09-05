import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title('K-means Clustering of Players Data')

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload the players dataset (CSV)", type="csv")

if uploaded_file is not None:
    players = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")

    # Select features for clustering
    features = ["overall", "potential", "wage_eur", "value_eur", "age"]
    
    # Drop rows with missing values in the selected features
    players = players.dropna(subset=features)
    data = players[features.copy()]

    st.write(data.head())
    st.write(data.describe())
    # Normalize the data between 1 and 10
    data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1
    
    # Slider for selecting number of clusters
    k = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=3)
    
    # K-means clustering functions
    def random_centroids(data, k):
        centroids = []
        for i in range(k):
            centroid = data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    def get_labels(data, centroids):
        distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
        return distances.idxmin(axis=1)

    def new_centroids(data, labels, k):
        return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

    def plot_clusters(data, labels, centroids, iteration):
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.transform(centroids.T)
        plt.figure()
        plt.title(f'Iteration {iteration}')
        plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
        plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color='red', marker='X')
        st.pyplot(plt)

    # Run K-means clustering
    max_iterations = 100
    centroids = random_centroids(data, k)
    old_centroids = pd.DataFrame()

    iteration = 1
    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels, k)
        
        st.write(f"Iteration {iteration}")
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1
    st.title('Centroids Data')
    st.write(centroids)
    st.title('Clustered data')
    st.write(players)

else:
    st.write("Please upload a CSV file to proceed.")
