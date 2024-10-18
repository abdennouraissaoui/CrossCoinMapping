def KMeanClustering(n_clusters=None,
                    price_scaled=None,
                   metric = "dtw",
                    max_iter=50
                   ):

    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html#tslearn.clustering.TimeSeriesKMeans

    n_clusters = n_clusters  # Specify the number of clusters
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, 
                          metric=metric, 
                          max_iter=max_iter, 
                          random_state=42)
    labels = kmeans.fit_predict(price_scaled)

    return labels


def kmeans_with_smape_ts(time_series_data, k, max_iter=10):
    """K-Means clustering for time series using SMAPE as the distance metric."""
    
    # Step 1: Initialize centroids randomly from time series data
    centroids = time_series_data[np.random.choice(len(time_series_data), k, replace=False)]
    
    for iteration in range(max_iter):
        clusters = [[] for _ in range(k)]  # Empty clusters
        
        # Step 2: Assign each time series to the cluster with the smallest SMAPE distance
        for series in time_series_data:
            distances = [smape(series, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(series)
        
        # Step 3: Update centroids by averaging the time series in each cluster
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                # Compute new centroid as the mean of time series in the cluster
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                # Handle empty cluster by reinitializing the centroid
                new_centroids.append(time_series_data[np.random.choice(len(time_series_data))])
        
        new_centroids = np.array(new_centroids)

        # Step 4: Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return clusters, centroids

def kmeans_with_min_distance(time_series_data, k, min_distance, max_iter=100):
    """
    K-Means clustering for time series with SMAPE as the distance metric 
    and a minimum distance constraint within clusters.
    """
    
    # Step 1: Initialize centroids randomly from time series data
    centroids = time_series_data[np.random.choice(len(time_series_data), k, replace=False)]
    
    # Array to store the label (cluster) for each time series
    labels = np.zeros(len(time_series_data), dtype=int)
    
    for iteration in range(max_iter):
        clusters = [[] for _ in range(k)]  # Empty clusters
        updated_labels = []  # To store labels after each iteration
        
        # Step 2: Assign each time series to the cluster with the smallest SMAPE distance
        for idx, series in enumerate(time_series_data):
            distances = [smape(series, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            
            # Ensure the distance within the cluster is below the minimum threshold
            too_far = False
            for existing_series in clusters[cluster_idx]:
                if smape(existing_series, series) > min_distance:
                    too_far = True
                    break

            if not too_far:
                clusters[cluster_idx].append(series)
                labels[idx] = cluster_idx  # Store the label for this time series
            else:
                # If the time series is too far from the points in the cluster, assign it to the closest valid cluster
                for other_cluster_idx in range(k):
                    if other_cluster_idx != cluster_idx:
                        distances = [smape(series, centroids[other_cluster_idx])]
                        if all(smape(series, s) <= min_distance for s in clusters[other_cluster_idx]):
                            clusters[other_cluster_idx].append(series)
                            labels[idx] = other_cluster_idx
                            break
        
        # Step 3: Update centroids by averaging the time series in each cluster
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                # Compute new centroid as the mean of time series in the cluster
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                # Handle empty cluster by reinitializing the centroid
                new_centroids.append(time_series_data[np.random.choice(len(time_series_data))])
        
        new_centroids = np.array(new_centroids)

        # Step 4: Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return labels, centroids