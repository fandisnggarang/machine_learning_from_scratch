import numpy as np 
import matplotlib.pyplot as plt 

class KMeans_Clustering():

    # parameter initialization 
    def __init__(self, k=5, num_iters=100, plot_steps=False, tol=1e-4, random_state=None, distance_measurement='euclidean'): 
        self.k         = k
        self.num_iters = num_iters
        self.plot_steps= plot_steps
        self.tol       = tol
        self.distance_measurement = distance_measurement

        # validate distance measurement input
        if distance_measurement not in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError("Invalid distance measurement. Choose from 'euclidean', 'manhattan', or 'cosine'.")

        if random_state: 
            np.random.seed(random_state)

        self.clusters  = [[] for _ in range(self.k)]

        self.centroids = []

    # formula of euclidean distance
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2)) 
    
    # formula of manhattan distance
    def manhattan_distance(self, x1, x2): 
        return np.sum(np.abs(x1-x2))
    
    # formula of cosine distance
    def cosine_distance(self, x1, x2): 
        dot_output = np.sum(x1 * x2)
        norm_x1    = np.sqrt(np.sum(x1 ** 2))
        norm_x2    = np.sqrt(np.sum(x2 ** 2))

        return 1 - (dot_output / (norm_x1 * norm_x2))

    # prediction process
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # choosing sample randomly and create centroid
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids     = [self.X[idx] for idx in random_sample_idxs]

        for i in range(self.num_iters):

            # assign samples to the nearest centroids to form clusters
            self.clusters = self.create_clusters(self.centroids)

            if self.plot_steps:
                print('Visualization after clusters are formed:')
                print()
                self.plot()

            # save old centroids to check convergence
            old_centroids = self.centroids 

            # calculate new centroids to check convergence
            self.centroids= self.update_centroid(self.clusters)

            # check for convergence, if centroids stop changing significantly
            if self.is_converged(old_centroids, self.centroids):
                break

            if self.plot_steps:
                print('Visualization after centroids have been updated:')
                print()
                self.plot()

        return self.get_label(self.clusters)

    # create clusters by assigning samples to the nearest centroid
    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X): 
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    # determine closest centroid to a sample
    def closest_centroid(self, sample, centroids):
        if self.distance_measurement == 'euclidean':
            distances   = [self.euclidean_distance(sample, point) for point in centroids]
        elif self.distance_measurement == 'manhattan': 
            distances   = [self.manhattan_distance(sample, point) for point in centroids]
        elif self.distance_measurement == 'cosine':
            distances   = [self.cosine_distance(sample, point) for point in centroids]
        closest_centroid_idx = np.argmin(distances) 

        return closest_centroid_idx

    # update centroid by calculating the mean of the points in each cluster
    def update_centroid(self, clusters): 
        centroids = np.zeros((self.k, self.n_features)) 
        for idx, cluster in enumerate(clusters):
            if len(cluster) == 0: 
                centroids[idx] = self.centroids[idx]
            else: 
                centroids[idx] = np.mean(self.X[cluster], axis = 0)

        return centroids 
    
    # check if centroids have converged
    def is_converged(self, old_centroids, centroids): 
        total_distance = 0
        for i in range(self.k):
            if self.distance_measurement == 'euclidean':
                total_distance += self.euclidean_distance(old_centroids[i], centroids[i])
            elif self.distance_measurement == 'manhattan': 
                total_distance += self.manhattan_distance(old_centroids[i], centroids[i])
            elif self.distance_measurement == 'cosine':
                total_distance += self.cosine_distance(old_centroids[i], centroids[i])

        return total_distance < self.tol
    
    # assign labels to each data point
    def get_label(self, clusters): 
        self.labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters): 
            for idx in cluster: 
                self.labels[idx] = cluster_idx
                 
        return self.labels       
    
    # plot the clustering result. But it is only for 2D data! 
    def plot(self):
        if self.n_features != 2:
            raise ValueError('Plotting is only able for 2D data')
        fig, ax = plt.subplots(figsize=(8, 4))

        # create cluster with different colors and mark the centroids with black 'x'
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

# Modified from Patric Loeber's code. 
# Check: https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch/blob/main/10%20KMeans/kmeans.py


