import sys
sys.path.append("../..")

from solution import solve
from optalg.example_nlps.quadratic_identity_2 import QuadraticIdentity2
from optalg.interface.nlp_traced import NLPTraced
import numpy as np

# You can freely modify this script to play around with your solver
# and the problems we provide in test.py
# Run the script with
# python3 play.py


# Example:
problem = NLPTraced(QuadraticIdentity2())
print("Funktionswert an einem Beispielpunkt (z.B. x = [1, 1]):", problem.evaluate(np.array([1, 1])))
x_dim = problem.getDimension()
#print('x_dim: ', x_dim)
x = solve(problem)
#print("found solution", x)
#print("real solution", np.zeros(2))
