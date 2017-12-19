from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([3,4,5,6,8,9])
ys = np.array([6,7,9,12,13,15])

def best_fit_sloap(xs, ys):
    m = (  ( (mean(xs) * mean(ys)) - mean(xs*ys) ) /
    ( (mean(xs)**2)  - mean(xs**2) )  )
    return m

m = best_fit_sloap(xs,ys)

print(m)

''' plt.scatter(xs,ys, m)
plt.show() '''