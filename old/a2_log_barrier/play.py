import sys
sys.path.append("/home/yuxiang/Documents/optalg22/optimization_algorithms_w22")
import numpy as np
from assignments.a2_log_barrier.solution_1 import solve
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuaterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.interface.nlp_traced import NLPTraced
from optalg.interface.objective_type import OT
import matplotlib.pyplot as plt
MAX_EVALUATE = 100000000
H = np.array([[1., -1.], [-1., 2.]])
g = np.array([-2., -6.])
Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
bineq = np.array([2., 2., 3.])
problem = NLPTraced(
    QuadraticProgram(
        H=H,
        g=g,
        Aineq=Aineq,
        bineq=bineq),
    max_evaluate=MAX_EVALUATE)
x = solve(problem)
#problem.getInitializationSample = lambda: np.zeros(2)
#problem.getInitializationSample = lambda: np.zeros(2)

#problem = NLPTraced(problem)
#solution = np.array([0.66667, 1.3333])
solution = np.array([0.1, 0.1])
print(x[0])
print("solution:", solution)
ind = range(len(problem.trace_x))
ind2 = range(len(x[1]))
ind3 = range(len(x[2]))
ind4 = range(len(x[0]))
val0 = [i[0] for i in problem.trace_x]
#val1 = [i[1] for i in problem.trace_x]
#val2 = [i[2] for i in problem.trace_x]
val3 = [i[0] for i in problem.trace_phi]
val4 = [i[3][1] for i in problem.trace_J]
val5 = [i[0][0] for i in problem.trace_J]
#val6 = [i[1][0] for i in problem.trace_J]
val7 = x[1]
#val8 = np.reshape([0],(-1,1))
#val9 = [i for i in x[0]]
plt.scatter(ind,val0, s=2, marker='^', c="b")
#plt.scatter(ind,val1, s=2, marker='^', c="r")
# # plt.scatter(ind,val2, s=2, marker='^', c="g")
#plt.scatter(ind,val3, s=2, marker='^', c="y")
# #plt.scatter(ind,val4, s=2, marker='^', c="m")
#plt.scatter(ind,val5, s=2, marker='^', c="k")
#plt.scatter(ind,val6, s=2, marker='^', c="c")
#plt.scatter(ind2,val7, s=2, marker='^', c="c")
#plt.scatter(ind3,val8, s=2, marker='^', c="r")
#plt.scatter(ind4,val9, s=2, marker='^', c="b")

plt.show()
#print(val3, "\n")
#print("\n", val5)

