import sys
sys.path.append("../..")
from solution import solve
from optalg.example_nlps.hole import Hole
from optalg.example_nlps.barrier import Barrier
from optalg.example_nlps.quadratic_identity_2 import QuadraticIdentity2
from optalg.interface.nlp_traced import NLPTraced
import unittest
import numpy as np

class testGradientDescent(unittest.TestCase):
    """
    test Gradient Descent Solver
    """

    def testQuadraticIdentity(self):

        problem = NLPTraced(QuadraticIdentity2())
        x = solve(problem)
        solution = np.zeros(2)
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testHole(self):

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
        solution = np.zeros(3)
        x = solve(problem)
        self.assertTrue(np.linalg.norm(x-solution) < 1e-3)

    def testBarrier(self):

        problem = NLPTraced(Barrier())
        solution = 0.01 * np.ones(2)
        x = solve(problem)
        self.assertTrue(np.linalg.norm(x-solution) < 1e-3)


# usage:
# print results in terminal
# python3 test.py
# store results in file
# python3 test.py out.log
if __name__ == "__main__":
    if len(sys.argv) == 2:
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
            runner = unittest.TextTestRunner(f, verbosity=2)
            unittest.main(testRunner=runner)
    else:
        unittest.main()
