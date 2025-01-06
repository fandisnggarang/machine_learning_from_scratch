from collections import Counter
import numpy as np

class KNearest_Neighbor_Classifier():

    # parameter initialization
    def __init__(self, k_value, distance_metric):
        self.k_value = k_value
        self.distance_metric = distance_metric

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
            raise ValueError('For distance metric, choose euclidean or manhattan')
    
    # identify k nearest neighbors of a given test data point
    def k_nearest(self, test_data):
        distance_data = []
        for train_point, label in zip(self.x1, self.y1): 
            distance = self.find_distance(train_point, test_data)
            distance_data.append((train_point, label, distance))
        distance_data.sort(key=lambda x:x[2])
        k_nearest_neighbors = distance_data[:self.k_value]
        
        return k_nearest_neighbors

    # assign predicted label
    def label_predict(self, test_data):
        neighbors = self.k_nearest(test_data)
        labels = []
        for data in neighbors: 
            labels.append(data[1])
        majority_label = Counter(labels).most_common(1)[0][0]

        return majority_label
    
    # prediction process
    def predict(self, x2):
        y_pred    = []
        for test_data in x2:
            prediction = self.label_predict(test_data)
            y_pred.append(prediction)
        return y_pred
    
# Modified from Siddhardhan's code.
# Chcek: https://www.youtube.com/watch?v=atErwG9lCa8
