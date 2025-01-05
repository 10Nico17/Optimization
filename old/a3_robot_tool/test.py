import unittest
import sys
import math
import numpy as np

sys.path.append("../..")
from optalg.utils.finite_diff import *
from solution import RobotTool


class testProblem(unittest.TestCase):

    problem = RobotTool

    def set_(self, problem):
        self.problem = problem

    def generateProblem(self):
        q0 = np.zeros(4)
        pr = np.array([.5, 2.0 / 3.0])
        l = .5
        problem = self.problem(q0, pr, l)
        return problem

    def generateProblem2(self):
        q0 = np.zeros(4)
        pr = np.array([0., 0.])
        l = .5
        problem = self.problem(q0, pr, l)
        return problem

    def testValue1(self):
        problem = self.generateProblem()
        # in this configuration, p = pr
        x = np.pi / 180.0 * np.array([90, -90, -90, 0.])
        phi, _ = problem.evaluate(x)
        c = problem.l * (x - problem.q0) @ (x - problem.q0)
        self.assertAlmostEqual(c, phi @phi)

    def testValue2(self):
        problem = self.generateProblem()
        # in this configuration, q = q0
        x = np.zeros(4)
        phi, _ = problem.evaluate(x)
        e = np.array([1.5 + 1. / 3. - .5, -2. / 3.])
        c = e @ e
        self.assertAlmostEqual(c, phi @phi)

    def testValue3(self):
        problem = self.generateProblem2()
        x = np.array([0., 0., 0., 1.])
        phi, _ = problem.evaluate(x)
        e = np.array([1 + .5 + 1. / 3 + 1, 0, 0, 0, math.sqrt(.5) * 1.])
        c = e @ e
        self.assertAlmostEqual(c, phi @phi)

    def testValue4(self):
        problem = self.generateProblem2()
        x = np.array([0., 0., math.pi / 2., 1.])
        phi, J = problem.evaluate(x)
        e = np.array([1 + .5, 1. / 3. + 1.0, 0, 0, math.sqrt(.5)
                      * math.pi / 2., math.sqrt(.5) * 1.])
        c = e @ e
        self.assertAlmostEqual(c, phi @phi)

    def testJacobian2(self):
        problem = self.generateProblem2()
        x = np.array([0., 0., 0., 1.])
        flag, _, _ = check_nlp(problem.evaluate, x, 1e-5)
        self.assertTrue(flag)

    def testJacobian(self):
        problem = self.generateProblem()
        x = np.pi / 180.0 * np.array([90., -90., -90., 0.])
        flag, _, _ = check_nlp(problem.evaluate, x, 1e-5)
        self.assertTrue(flag)


# Run tests with:
# python3 test.py
# Too see help and options
# python3 test.py --help
if __name__ == "__main__":
    unittest.main()
