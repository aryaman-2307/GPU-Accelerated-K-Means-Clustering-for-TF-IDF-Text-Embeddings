import cupy as cp

def kmeans_gpu(X, k, max_iter=100):
    n_samples, n_features = X.shape
    indices = cp.random.choice(n_samples, k, replace=False)
    centroids = X[indices]
    
    for _ in range(max_iter):
        # Compute distances and assign clusters
        dists = cp.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = cp.argmin(dists, axis=1)

        # Update centroids
        new_centroids = cp.vstack([
            X[labels == i].mean(axis=0) if cp.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if cp.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels.get(), centroids.get()
