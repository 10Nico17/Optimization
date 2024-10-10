# First let us import all the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as au
from autograd import grad, jacobian
import scipy
from scipy.optimize import line_search


def himm(x): # Objective function
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

grad_himm = grad(himm) # Gradient of the objective function

x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
z = np.zeros(([len(x), len(y)]))
for i in range(0, len(x)):
    for j in range(0, len(y)):
        z[j, i] = himm([x[i], y[j]])

contours=plt.contour(x, y, z, 100, cmap=plt.cm.gnuplot)
plt.clabel(contours, inline=1, fontsize=10)
plt.show()



start_point = np.array([-2.5, 2.8])
delta = np.array([-2.5, -1])
alpha_1 = 10**-4
alpha_2 = 0.325
beta_max = 0.6

res=line_search(f = himm, myfprime = grad_himm, xk = start_point, pk = delta, c1 = alpha_1, c2 = alpha_2, amax = beta_max)
print('Result: ', res)