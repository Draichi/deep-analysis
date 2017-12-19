from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([3,4,5,6,8,9])
ys = np.array([6,7,9,12,13,15])

def best_fit_sloap_and_intercept(xs, ys):
    m = (  ( (mean(xs) * mean(ys)) - mean(xs*ys) ) /
    ( (mean(xs)**2)  - mean(xs**2) )  )

    b = mean(ys) - ( m * mean(xs) )
    return m, b

m, b = best_fit_sloap_and_intercept(xs,ys)

regression_line = [ (m*x)+b for x in xs ]
# This above is same as: 
# for x in xs:
#   regression_line.append((m*x)+b)

predict_x = 11
predict_y = (m*predict_x)+b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y)
plt.plot(xs, regression_line)
plt.show()