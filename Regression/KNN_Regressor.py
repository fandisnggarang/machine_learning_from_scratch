import numpy as np
from collections import Counter

class KNearest_Neighbors_Regressor():

    # parameter initialization
    def __init__(self, k_value, distance_metric, average_metric):
        self.k_value         = k_value
        self.distance_metric = distance_metric
        self.average_metric  = average_metric

    # fitting process
    def fit(self, x1, y1):
        self.x1 = x1
        self.y1 = y1

    # distance metric with euclidean or manhattan formula
    def find_distance(self, train_data, test_data):
        if self.distance_metric == 'euclidean': 
            return np.sqrt(np.sum((train_data - test_data) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(train_data - test_data))
        else:
            raise ValueError("For distance metric, choose 'euclidean' or 'manhattan'")
    
    # identify k nearest neighbors of a given test data point
    def k_nearest(self, test_data):
        distance_data = []
        for train_point, target_value in zip(self.x1, self.y1): 
            distance = self.find_distance(train_point, test_data)
            distance_data.append((train_point, target_value, distance))
        distance_data.sort(key=lambda x:x[2])
        k_nearest_neighbors = distance_data[:self.k_value]
        
        return k_nearest_neighbors

    # identify distance average using simple or weighted mean
    def distance_average(self, test_data):
        y_target = []
        k_nearest_neighbors = self.k_nearest(test_data)
        for train_point, target_value, distance in k_nearest_neighbors:
            y_target.append(target_value)
    
        weights= []
        if self.average_metric == 'simple mean':
            return np.mean(y_target)
        elif self.average_metric == 'weighted mean':
            for _, _, distance in k_nearest_neighbors:
                count = 1/(distance + 1e-5)
                weights.append(count)
            weighted_sum = sum(target * weight for (_, target, distance), weight in zip(k_nearest_neighbors, weights))
            return weighted_sum/sum(weights)
        else:
            raise ValueError("Please choose 'simple mean' or 'weighted mean' for average_metric")

    # prediction process
    def predict(self, x2):
        y_pred    = []
        for test_data in x2:
            prediction = self.distance_average(test_data)
            y_pred.append(prediction)
        return y_pred
    
# Adapted from Siddhardhan's KNN Classifier code.
# Check: https://www.youtube.com/watch?v=atErwG9lCa8
