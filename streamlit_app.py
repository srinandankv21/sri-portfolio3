import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title('Generic K-means Clustering App')

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload any dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")

    # Display the first few rows of the dataset
    st.write(data.head())

    # Filter numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check if there are any numeric columns
    if len(numeric_columns) > 0:
        st.write("Select the features to use for clustering:")
        
        # Multi-select box for choosing the features
        selected_features = st.multiselect("Features", numeric_columns, default=numeric_columns[:5])
        
        if selected_features:
            data = data[selected_features].dropna()
            
            # Normalize the data between 1 and 10
            data_normalized = ((data - data.min()) / (data.max() - data.min())) * 9 + 1
            
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

            def plot_clusters(data, labels, centroids):
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(data)
                centroids_2d = pca.transform(centroids.T)
                plt.figure()
                plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
                plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color='red', marker='X')
                return plt

            # Run K-means clustering
            max_iterations = 100
            centroids = random_centroids(data_normalized, k)
            old_centroids = pd.DataFrame()

            images = []
            all_labels = None

            iteration = 1
            while iteration <= max_iterations and not centroids.equals(old_centroids):
                old_centroids = centroids
                labels = get_labels(data_normalized, centroids)
                all_labels = labels  # Save the final labels after the last iteration
                centroids = new_centroids(data_normalized, labels, k)
                fig = plot_clusters(data_normalized, labels, centroids)
                st.pyplot(fig)
                iteration += 1
            
           
            
            
            # Show final clustered data with cluster labels
            if all_labels is not None:
                st.subheader('Clustered Data with Labels')
                clustered_data = data.copy()
                clustered_data['Cluster'] = all_labels.values
                st.write(clustered_data)
        else:
            st.write("Please select at least one feature to proceed with clustering.")
    else:
        st.write("The dataset doesn't have any numeric columns for clustering.")
else:
    st.write("Please upload a CSV file to proceed.")
