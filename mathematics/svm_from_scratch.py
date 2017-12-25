import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'b', -1:'r'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w, b] }
        opt_dict = {}
        # each time we have a vector we want
        # to tranform by this values
        transforms = [[1,1], [-1,1], [-1,-1],[1,-1]]
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        # we dont want to hold this on memory
        all_data = None
        # big steps first
        setp_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # ]
                      # point of expense:
                      self.max_feature_value * 0.001]
        # 5 = extremely expensive
        b_range_multiple = 5
        # we dont need to take as small
        # of steps with b as we do w
        b_multiple = 5
        # fist element in vector w
        latest_optimum = self.max_feature_value*10
        for step in setp_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # will stay false until we dont 
            # have more steps to take.
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple):
                    for trasformation in transforms:
                        # w tranformed
                        w_t = w*trasformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        for i in self.data:
                            # self.data is a dictionary
                            for xi in self.data[i]:
                                # yi(xi.w+b) >= 1
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    # if one of the samples does not fit
                                    # this definition, the whole thing is throw now
                                    found_option = False
                        # if everything check out
                        if found_option:
                            # np.linalg.norm(w_t) = vector magnitude
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    # w = [5,5]
                    # step = 1
                    # w - step = [4,4] 
                    w = w - step
            # we're sorting a list of magnitudes
            # from lowest to highest
            norms = sorted([n for n in opt_dict])
            # smallest norm
            opt_choice = opt_dict[norms[0]]
            # ||w|| : [w,b]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2
            
        ############
        # this is just to print the optimization steps
        for i in self.data:
            # self.data is a dictionary
            for xi in self.data[i]:
                # yi(xi.w+b) >= 1
                yi = i
                print(xi,':',yi*(np.dot(self.w, xi) + self.b))
        ############

    def predict(self, features):
        # sign ( Xi.w + b )
        classification = np.sign( np.dot( np.array(features), self.w ) + self.b )
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification
    
    # this vizualize function is important only for humans
    def vizualize(self):
        [[self.ax.scatter(x[0], x[1], s=200, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        # v = x.w+b
        # positive support vector = 1
        # negative support vector = -1
        # decision = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1,1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        # (w.x+b) = 1
        # positive support vaector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        # 'k' is black
        self.ax.plot([hyp_x_min, hyp_x_max],[psv1, psv2], 'k')
        # (w.x+b) = -1
        # negative support vaector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max],[nsv1, nsv2], 'k')
        # (w.x+b) = 0
        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        # 'y--' is yellow dashed
        self.ax.plot([hyp_x_min, hyp_x_max],[db1, db2], 'y--')
        #show
        plt.show()

data_dict = {-1:np.array([ [1,7], [2,8], [3,8] ]), 
            1:np.array([ [9,1], [6,-1], [7,0] ])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10], [2,7], [4,5], [5,6], [4,5], [3,-1], [6,0], [8,-2], [5,8]]
for p in predict_us:
    svm.predict(p)

svm.vizualize()

# support vectors yi(xi.w+b) = 1
# u'll know that u've found a really great value
# for w and b when both your positives and negatives classes
# have a value that is close to 1.

# if u want to reach a better number,
# add more step_sizes