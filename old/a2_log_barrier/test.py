import sys
sys.path.append("../..")
import unittest
import numpy as np
from solution_1 import solve
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuaterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.interface.nlp_traced import NLPTraced

MAX_EVALUATE = 10000


class testLogBarrier(unittest.TestCase):

    def testLinear1(self):

        problem = NLPTraced(LinearProgramIneq(2), max_evaluate=MAX_EVALUATE)
        x = solve(problem)
        solution = np.zeros(2)
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testLinear2(self):

        problem = NLPTraced(LinearProgramIneq(20), max_evaluate=MAX_EVALUATE)
        problem.getInitializationSample = lambda: .1 + .1 * np.arange(20)
        x = solve(problem)
        solution = np.zeros(20)
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testQuadraticIneq(self):
        """
        """
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

        problem.getInitializationSample = lambda: np.zeros(2)
        x = solve(problem)
        solution = np.array([0.6667, 1.3333])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testHalfcircle(self):
        problem = NLPTraced(HalfCircle(), max_evaluate=MAX_EVALUATE)
        x = solve(problem)
        solution = np.array([0, -1.])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testQuaterCircle(self):
        problem = NLPTraced(QuaterCircle(), max_evaluate=MAX_EVALUATE)
        x = solve(problem)
        solution = np.array([0, 0.])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testQuadraticIneq2(self):
        """
        """
        H = np.array([[10., 0.], [0., 1.]])
        g = np.array([1., 1.])
        Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
        bineq = np.array([2., 2., 3.])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=MAX_EVALUATE)

        problem.getInitializationSample = lambda: np.zeros(2)
        x = solve(problem)
        solution = np.array([-0.1, -1])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testQuadraticIneq3(self):
        """
        """
        H = np.array([[100000., 0.], [0., 1.]])
        g = np.array([0., 0.])
        Aineq = np.array([[-1., 0], [0., -1.]])
        bineq = np.array([-0.1, -0.1])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=MAX_EVALUATE)

        problem.getInitializationSample = lambda: 4 * np.ones(2)
        x = solve(problem)
        solution = np.array([0.1, 0.1])

        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)


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
