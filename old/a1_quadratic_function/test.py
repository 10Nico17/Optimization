import sys
sys.path.append("../..")

from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess
import numpy as np
import unittest
from solution import NLP_xCCx


# import the test classes


class test_NLP_xCCx(unittest.TestCase):
    """
    test NLP_xCCx
    """

    def testValue(self):

        C = np.ones((2, 2))
        problem = NLP_xCCx(C)
        x = np.ones(2)
        value = problem.evaluate(x)[0][0]
        solution = 8
        self.assertAlmostEqual(value, solution)

    def testJacobian(self):

        C = np.ones((2, 2))
        problem = NLP_xCCx(C)
        x = np.array([-1, .5])
        _, J = problem.evaluate(x)
        eps = 1e-5
        Jdiff = finite_diff_J(problem, x, eps)
        self.assertTrue(np.allclose(J, Jdiff, eps * 10))

    def testHessian(self):

        C = np.ones((2, 2))
        problem = NLP_xCCx(C)
        x = np.array([-1, .1])
        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        self.assertTrue(np.allclose(H, Hdiff, 10 * tol, 10 * tol))


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
