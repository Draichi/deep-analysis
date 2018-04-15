import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=50, centers=3, n_features=2)

#X = np.array([[1,2],[1.5,1.8],[5,8],[8,2],[9,1],[10,3],[8,8],[1,0.6],[9,11]])

''' plt.scatter(X[:,0], X[:,1], s=150)
plt.show() '''

colors = 10*["g","r","b","k","o","p"]

class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
            # we gonna find the centroid of all our data
            all_data_centroid = np.average(data, axis=0)
            # this will give us the magnitude of that origin of data centroid
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}
        for i in range(len(data)):
            # we're giving the centroid an id (key)
            # and the value 'll be the data
            centroids[i] = data[i]
        # from 0 to 99 (radius_norm_step=100)
        # revese the order of this list with ::-1
        # now is 99 to 0
        weights = [i for i in range(self.radius_norm_step)][::-1]
        while True:
            new_centroids = []
            for i in centroids:
                # this'll be populated with all featuresets
                # who are in our bandwith
                in_bandwith = []        
                centroid = centroids[i]
                for featureset in data:
                    # full distance
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        # this is just for when the featureset is comparing
                        # distances to itself
                        distance = 0.00000001
                    # this is how many radius steps we took
                    # more steps we take, less that weight should be
                    weight_index  =int(distance / self.radius)
                    # if the distance is greater than max distance,
                    # the weight index would be that max distance
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    to_add = (weights[weight_index]**2)*[featureset]
                    # this is a list plus a list
                    in_bandwith += to_add
                # this give us the mean vector of all of our vectors
                new_centroid = np.average(in_bandwith, axis=0)
                # we're converting an array to a tuple
                new_centroids.append(tuple(new_centroid))
            # and this is why we're using tuples,
            # set of tuples to get unique elements
            uniques = sorted(list(set(new_centroids)))
            
            to_pop = []
            for i in uniques:
                for ii in uniques:
                    # the centroids will be really close to each other
                    # we dont need .00...01 different to each other
                    # if it's itself we pass
                    if i == ii:
                        pass
                    # if the distance between this 2 vectors is less or equal the radius,
                    # if they are in one radius step from each other,
                    # this need to be converged to be the same centroid
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

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
        
        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)
    
    def predict(self, data):
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        classifications = distances.index(min(distances))
        return classification

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    # we gonna use classification as an index
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()