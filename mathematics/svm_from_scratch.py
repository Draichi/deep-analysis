import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')



class Support_Vector_Machine:

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
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
                      # point of expense:
                      self.max_feature_value * 0.001]
        # extremely expensive
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

    def predict(self, features):
        # sign ( Xi.w + b )
        classification = np.sign( np.dot( np.array(features), self.w ) + self.b )
        return classification



data_dict = {-1:np.array([ [1,7], [2,8], [3,8] ]), 
            1:np.array([ [5,1], [6,-1], [7,3] ])}

# support vectors yi(xi.w+b) = 1
# u'll know that u've found a really great value
# for w and b when both your positives and negatives classes
# have a value that is close to 1.

# if u want to reach a better number,
# add more step_sizes