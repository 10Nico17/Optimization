import sys
sys.path.append("../..")
import unittest
import numpy as np
from solution import solve
from optalg.example_nlps.hole import Hole
from optalg.example_nlps.barrier import Barrier
from optalg.example_nlps.rosenbrock import Rosenbrock
from optalg.example_nlps.rosenbrockN import RosenbrockN
from optalg.example_nlps.cos_lsqs import Cos_lsqs
from optalg.example_nlps.quadratic import Quadratic
from optalg.interface.nlp_traced import NLPTraced


class testSolverUnconstrained(unittest.TestCase):
    """
    """

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

        C = make_C_exercise1(3, .01)
        problem = NLPTraced(Hole(C, 1.5))
        solution = np.zeros(3)
        x = solve(problem)
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testBarrier(self):

        problem = NLPTraced(Barrier())
        solution = 0.01 * np.ones(2)
        x = solve(problem)
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testRosenbrock_easy(self):

        problem = NLPTraced(Rosenbrock(1., 1.5))

        problem.getInitializationSample = lambda: np.array([1.2, 1.2])

        x = solve(problem)

        solution = np.array([1., 1.])
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testQuadratic(self):
        def make_C_exercise1(n, c):
            """
            n: integer
            c: float
            """
            C = np.zeros((n, n))
            for i in range(n):
                C[i, i] = c ** (float(i - 1) / (n - 1))
            return C

        problem = NLPTraced(
            Quadratic(make_C_exercise1(10, .00009)), max_evaluate=1000)

        x = solve(problem)
        solution = np.zeros(problem.getDimension())
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testBarrier_large(self):

        n = 10
        problem = NLPTraced(Barrier(n=n))
        problem.getInitializationSample = lambda: np.ones(
            n) + 0.1 * np.arange(n)
        solution = 0.01 * np.ones(n)

        x = solve(problem)

        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testRosenbrock(self):
        problem = NLPTraced(Rosenbrock(2, 100))
        solution = np.array([2, 4])
        x = solve(problem)
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testRosenbrockN(self):
        N = 7
        problem = NLPTraced(RosenbrockN(N))
        x = solve(problem)
        solution = np.ones(N)
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testCosLsqs(self):

        A = .1 * np.array([[1., 2., 3.], [4, 5, 6]])
        b = np.zeros(2)
        problem = NLPTraced(Cos_lsqs(A, b))
        x = solve(problem)
        solution = np.zeros(3)
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)


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
