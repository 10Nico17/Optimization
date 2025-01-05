import sys
sys.path.append("../..")
import numpy as np
from solution import solve
from optalg.example_nlps.hole import Hole
from optalg.example_nlps.barrier import Barrier
from optalg.example_nlps.rosenbrock import Rosenbrock
from optalg.example_nlps.rosenbrockN import RosenbrockN
from optalg.example_nlps.cos_lsqs import Cos_lsqs
from optalg.example_nlps.quadratic import Quadratic
from optalg.interface.nlp_traced import NLPTraced
import matplotlib.pyplot as plt

def make_C_exercise1(n, c):
    """
    n: integer
    c: float
    """
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = c ** (float(i - 1) / (n - 1))
    return C

A = np.random.rand(10,10)
C = make_C_exercise1(3, .01)
problem = NLPTraced(Hole(C, 1.5))
problem = NLPTraced(Barrier(2,0.01))
problem = NLPTraced(Quadratic(A))
solution = np.zeros(3)
x = solve(problem)


print(x)
#print(problem.trace_x)
ind = range(len(problem.trace_x))
val0 = [i[0] for i in problem.trace_x]
# val1 = [i[1] for i in problem.trace_x]
# val2 = [i[2] for i in problem.trace_x]
# val3 = problem.trace_phi
val4 = [i[0][0] for i in problem.trace_J]
# val5 = [i[0][1] for i in problem.trace_J]
# val6 = [i[0][2] for i in problem.trace_J]
#plt.scatter(ind,val0, s=2, marker='^', c="b")
# plt.scatter(ind,val1, s=2, marker='^', c="r")
# plt.scatter(ind,val2, s=2, marker='^', c="g")
# plt.scatter(ind,val3, s=2, marker='^', c="y")
plt.scatter(ind,val4, s=2, marker='^', c="m")
# #plt.scatter(ind,val5, s=2, marker='^', c="k")
# #plt.scatter(ind,val6, s=2, marker='^', c="c")
plt.show()
# print(np.linalg.inv(problem.getFHessian(problem.getInitializationSample()) + np.eye(3))@(problem.evaluate(problem.getInitializationSample())[1][0]))
# #print(problem.evaluate(problem.getInitializationSample())[1][0])
#print(solution)
print(problem.counter_evaluate)
#print(val4)
# print(np.shape(problem.getFHessian(problem.getInitializationSample()))[0])