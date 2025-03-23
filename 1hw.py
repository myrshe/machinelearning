import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# метод локтя

def elbow_method(data):
    inertia = []
    cluster_range = range(1, 11)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    plt.plot(cluster_range, inertia, marker='o')
    plt.title('Метод локтя')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.show()


# k-means

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)


def update_centroids(data, clusters, k):
    centroids = []
    for i in range(k):
        points = data[clusters == i]
        if len(points) > 0:
            centroid = np.mean(points, axis=0)
        else:
            centroid = data[np.random.choice(len(data))]
        centroids.append(centroid)
    return np.array(centroids)


def plot_clusters(data, clusters, centroids, step):
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    for i in range(len(centroids)):
        points = data[clusters == i]
        plt.scatter(points[:, 0], points[:, 1], color=colors[i % len(colors)], label=f'Кластер {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Центроиды')
    plt.title(f'Шаг {step}')
    plt.legend()
    plt.show()

def custom_kmeans(data, k, max_steps=10):
    np.random.seed(0)
    random_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[random_indices]
    for step in range(max_steps):
        clusters = assign_clusters(data, centroids)
        plot_clusters(data, clusters, centroids, step + 1)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):

            break
        centroids = new_centroids



def main():
    irises = load_iris()
    data = irises.data[:, :2]  # используется только два признака


    elbow_method(data)

    custom_kmeans(data, k=3)


if __name__ == '__main__':
    main()
