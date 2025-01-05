import sys
sys.path.append("../..")
import unittest
import numpy as np
from solution import solve
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.example_nlps.logistic_bounds import LogisticWithBounds
from optalg.example_nlps.nonlinearA import NonlinearA
from optalg.interface.nlp_traced import NLPTraced
from optalg.example_nlps.f_r_eq import F_R_Eq


FACTOR = 30


def _solve(nlp):
    return solve(nlp)


class testAuglag(unittest.TestCase):

    def testLinear1(self):

        problem = NLPTraced(LinearProgramIneq(2), max_evaluate=39 * FACTOR)
        x = _solve(problem)
        solution = np.zeros(2)
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
            max_evaluate=54 * FACTOR)

        problem.getInitializationSample = lambda: np.zeros(2)
        x = _solve(problem)
        solution = np.array([0.6667, 1.3333])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testHalfcircle(self):
        problem = NLPTraced(HalfCircle(), max_evaluate=64 * FACTOR)
        x = _solve(problem)
        solution = np.array([0, -1.])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)


    def testQuadraticIneq3(self):
        """
        """
        H = np.array([[100., 0.], [0., 1.]])
        g = np.array([0., 0.])
        Aineq = np.array([[-1., 0], [0., -1.]])
        bineq = np.array([-0.1, -0.1])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=320 * FACTOR)

        problem.getInitializationSample = lambda: 4 * np.ones(2)
        x = _solve(problem)
        solution = np.array([0.1, 0.1])

        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testLogisticBounds(self):
        problem = NLPTraced(
            LogisticWithBounds(), 60 * FACTOR)
        x = _solve(problem)
        solution = np.array([2, 2, 1.0369])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testQuadraticB(self):
        n = 3
        H = np.array([[1., -1., 1], [-1, 2, -2], [1, -2, 4]])
        g = np.array([2, -3, 1])
        Aineq = np.vstack((np.identity(n), -np.identity(n)))
        bineq = np.concatenate((np.ones(n), np.zeros(n)))
        Aeq = np.ones(3).reshape(1, -1)
        beq = np.array([.5])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aeq=Aeq,
                beq=beq,
                Aineq=Aineq,
                bineq=bineq),
            142 * FACTOR)
        x = solve(problem)
        solution = np.array([0, 0.5, 0])
        self.assertTrue(np.linalg.norm(x - solution) < .01)

    def test_nonlinearA(self):
        problem = NLPTraced(
            NonlinearA(), 307 * FACTOR)
        x = solve(problem)
        solution = np.array([1.00000000, 0])
        self.assertTrue(np.linalg.norm(solution - x) < .01)

    def test_f_r(self):

        Q = np.array([[1000., 0.], [0., 1.]])
        R = np.ones((2, 2))
        d = np.zeros(2)
        A = np.array([[1., 1.], [1., 0.]])
        b = np.zeros(2)

        problem = NLPTraced(F_R_Eq(Q, R, d, A, b), 21 * FACTOR)
        x = solve(problem)
        solution = np.array([0., 0.])
        self.assertTrue(np.linalg.norm(x - solution) < .01)


# Run tests with:
# python3 test.py

# Too see help and options
# python3 test.py --help

if __name__ == "__main__":
    unittest.main()
