import sys
sys.path.append("../..")
from solution import NLP_nonlinear
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess
import numpy as np
import unittest
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess



# You can freely modify this script to play around with 
# the implementation of your NLP

# Example
C = np.random.rand(50, 10)
#C = np.ones((2, 2))
problem = NLP_nonlinear(C)
x = np.random.rand(10)
#x = np.array([-1, .1, .6])
#x = np.array([1, -1, -0.5])
y, J = problem.evaluate(x)
value = y[0]
jacob = J[0]
solution = 1. / 8.
eps = 1e-5
Jdiff = finite_diff_J(problem, x, eps)
def f(x):
    return problem.evaluate(x)[0][0]
tol = 1e-5
Hdiff = finite_diff_hess(f, x, tol)
H = problem.getFHessian(x)
eps = 1e-5
diff = finite_diff_J(problem, x, eps)
CC = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12],[13,14,15,16]])
xx = np.array([1,-1,.5,2])
DD = CC.T@xx
#print(x)
#print(np.transpose(x))
#print(np.transpose(C))
print("found jacob matrix:","\n", jacob, "\n")
print("solution jacob matrix:","\n", Jdiff, "\n")
#print(diff)
#print("found Hessian matrix:","\n",H, "\n")
#print("solution Hessian matrix:","\n", Hdiff, "\n")
print(value)
#print("found solution", value, "\n")
#print(DD)
#print("real solution", solution)