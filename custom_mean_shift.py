import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],[1.5,1.8],[5,8],[8,2],[9,1],[10,3],[8,8],[1,0.6],[9,11]])

''' plt.scatter(X[:,0], X[:,1], s=150)
plt.show() '''

colors = ["g","r","b","k","o","p"]

class Mean_Shift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centroids = {}
        for i in range(len(data)):
            # we're giving the centroid an id (key)
            # and the value 'll be the data
            centroids[i] = data[i]
        while True:
            new_centroids = []
            for i in centroids:
                # this'll be populated with all featuresets
                # who are in our bandwith
                in_bandwith = []        
                centroid = centroids[i]
                for featureset in data:
                    # if that euclidian distance is less than self.radius
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwith.append(featureset)
                # this give us the mean vector of all of our vectors
                new_centroid = np.average(in_bandwith, axis=0)
                # we're converting an array to a tuple
                new_centroids.append(tuple(new_centroid))
            # and this is why we're using tuples,
            # set of tuples to get unique elements
            uniques = sorted(list(set(new_centroids)))
            # just copying the centroids dictonary without taking the attributies
            prev_centroids = dict(centroids)
            # we're cleaning the centroids dict to populated
            # it again with an array version
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            optimized = True
            for i in centroids:
                # we're just comparing the arrays to se if they're equal, if they're not:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break
        # once we're all done we gonna reset our centroids
        self.centroids = centroids
    
    def predict(self, data):
        pass

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:,0], X[:,1], s=150)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()