import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

''' plt.scatter(X[:,0], X[:,1], s=150)
plt.show() '''

colors = ["g","r","b","k","o","p"]

class K_Means:
    # tolerance is how much that centroid is gonna move by % change
    # max iteration is to limit the number of cycles we're willing to run 
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        # we're iterating into data (X in this case)
        # the first 2 centroids (k=2) is gonna be [1,2],[1.5,1.8]
        for i in range(self.k):
            self.centroids[i] = data[i]
        # optimization process starts here
        # we start with empty classifications
        for i in range(self.max_iter):
            self.classifications = {}
            # and then create 2 dict keys (range of k)
            for i in range(self.k):
                self.classifications[i] = []
            # this is creating a list that is been populated with
            # K number values, because of centroids in self.centroids that
            # containes k number of centroids (0 and 1)
            # 0 index of this list will be the distance to the 0 with centroid
            # the first element will be the distance from that datapoint to the centroid 1
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                # whats the index values of minimum of distances
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            # here we're trying to compare the centroids so we can
            # find how much they've changed, so we can use that tolerance value
            prev_centroids = dict(self.centroids)

            # this is gona take the average of all classifications that we have
            # and redefine the centroids
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            #if the centroids move more than tolerance, we're not optimized
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    # this shows how many iterations we went through and how big they are
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False
            
            # if it's optimized will break the for loop and stop us to run thro
            # every single one max_iter
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)


unknows = np.array([[1,2], [2,8], [3,4], [6,6], [12,12]])
for unknow in unknows:
    classification = clf.predict(unknow)
    plt.scatter(unknow[0], unknow[1], marker='*', color=colors[classification], s=30, linewidths=3)


plt.show()