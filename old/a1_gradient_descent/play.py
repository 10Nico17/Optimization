import sys
sys.path.append("C:\OptAlg22\optimization_algorithms_w22")

from solution import solve
from optalg.example_nlps.quadratic_identity_2 import QuadraticIdentity2
from optalg.example_nlps.hole import Hole
from optalg.interface.nlp_traced import NLPTraced
import numpy as np

# You can freely modify this script to play around with your solver
# and the problems we provide in test.py
# Run the script with
# python3 play.py


# Example: 
def make_C_exercise1(n, c):
    """
    n: integer
    c: float
    """
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = c ** (float(i - 1) / (n - 1))
    return C

C = make_C_exercise1(3, .1)
problem = NLPTraced(Hole(C, 1.5))
x= solve(problem)
co = problem.counter_evaluate
print("found solution", x)
print("real solution", np.zeros(3))
print("iterate num:", co)

