import sys
sys.path.append("../..")
import numpy as np

from solution import NLP_Gaussina_ineq
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess


# You can freely modify this script to play around with
# the implementation of your NLP


# Example:
D = np.ones((2, 2))
D = np.array([[1, 2],[3, -1]])
A = np.ones((2, 2))
b = np.ones(2)
x0 = np.zeros(2)

D = np.eye(2)
A = np.ones((3, 2))
b = np.array([2., 2., 2.])
x0 = np.array([1., .1])

problem = NLP_Gaussina_ineq(x0, D, A, b)
x = np.ones(2)
y, J = problem.evaluate(x)
H = problem.getFHessian(x)
solution = np.array([- np.exp(-2), -2., -2., -2.])
eps = 1e-5
Jdiff = finite_diff_J(problem, x, eps)
tol = 1e-4
def f(x):
    return problem.evaluate(x)[0][0]
Hdiff = finite_diff_hess(f, x, tol)
print("value:\n",y, "\n")
print("Jacob:\n",J, "\n")
print("Jacob solution:\n",Jdiff, "\n")
print("Hess:\n",H, "\n")
print("Hess solution:\n",Hdiff, "\n")
print(solution)

# Example:
D = np.ones((2, 2))
#D = np.array([[0.5, 2],[3, -1]])
A = np.ones((2, 2))
b = np.ones(2)
x0 = np.zeros(2)

D = np.eye(2)
A = np.ones((3, 2))
b = np.array([2., 2., 2.])
x0 = np.array([1., .1])
x0 = np.ones(2)
problem = NLP_Gaussina_ineq(x0, D, A, b)
x = np.zeros(2)
y, J = problem.evaluate(x)
pp = problem.getFHessian(x)
solution = np.array([- np.exp(-2), -2., -2., -2.])
#print(pp)

l1 = np.array([[4],[4]])
l2 = np.array([[4,4]])
#print(l1@l2)
print(y)
print(solution)