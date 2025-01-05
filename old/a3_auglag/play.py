import sys
sys.path.append("/home/yuxiang/Documents/optalg22/optimization_algorithms_w22")
import numpy as np
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuaterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.example_nlps.logistic_bounds import LogisticWithBounds
from optalg.example_nlps.nonlinearA import NonlinearA
from optalg.interface.nlp_traced import NLPTraced
from optalg.example_nlps.f_r_eq import F_R_Eq
from solution import *
from optalg.utils.finite_diff import *
from plot_tool import plotFunc
import matplotlib.pyplot as plt
# problem = LinearProgramIneq(2)
# x, trace_x, trace_lambda = solve(problem)
# trace_x = np.array(trace_x)
# trace_lambda = np.array(trace_lambda)
# #trace_x = np.transpose(trace_x)
# print(trace_x)
# solution = np.zeros(2)

FACTOR = 30
problem = NLPTraced(LinearProgramIneq(2), max_evaluate=39 * FACTOR)
# x, trace_x, trace_lambda = solve(problem)
x = solve(problem)
# solution = np.array([0.1, 0.1])
# trace_x = np.array(trace_x)
# trace_lambda = np.array(trace_lambda)
# print("x", x)
# print("solution", solution)
# value = trace_x[:,0]*2
# value_annotate = range(len(value))
# value_l = trace_lambda[:,0]
# for i,v in enumerate(value_annotate):
#     plt.annotate(str(v),(trace_x[:,0][i], value[i]))
#     #plt.annotate(str(v),(trace_x[:,0][i], value_l[i]))
# plt.scatter(trace_x[:,0],value, s=2, marker='^', c="b")
# #plt.scatter(trace_x[:,0],value_l, s=2, marker='^', c="r")
plotFunc(problem.evaluate, [-2,-2], [2,2])
# plt.show()
