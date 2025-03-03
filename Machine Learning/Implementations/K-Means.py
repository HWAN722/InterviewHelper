import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)

    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        # print(distances)
        assignments = np.argmin(distances, axis=0)
        # print(assignments)

        # Apply Means
        new_centroids = np.array(
            [points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 else centroids[i] for i in
             range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids, 4)
    return [tuple(centroid) for centroid in centroids]


print(k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2,
                         [(1, 1, 1), (10, 10, 10)], 1))
