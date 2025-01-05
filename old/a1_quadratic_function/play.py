import sys
sys.path.append("../..")

from solution import NLP_xCCx
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess
import numpy as np
import unittest

import optalg.utils.finite_diff as fd
# You can freely modify this script to play around with 
# the implementation of your NLP


# Example:
C = np.random.rand(7,6)

x = np.random.rand(6)
x_t = np.transpose(x)
C_t = np.transpose(C)
eps = 1e-4

problem = NLP_xCCx(C)
def f2(x):
    return problem.evaluate(x)[0][0]
#def f(x):
#    y = x@C@C_t@x_t
#    return y

#H = fd.finite_diff_hess(f, x, eps)
H2 = fd.finite_diff_hess(f2, x, eps)
H3 = problem.getFHessian(x)
J = problem.evaluate(x)[1][0]
y = problem.evaluate(x)[0][0]
eps = 1e-5
J_dif = finite_diff_J(problem, x, eps)
print(y)
#print(f2(x))
print("found jacob martix:\n", J,"\n",
        "solution jacob matrix:\n", J_dif,"\n",
        "found Hessian matrix:\n", H3,"\n",
        "solution Hessian matrix:\n", H2)
#print(J, "\n", J_dif)