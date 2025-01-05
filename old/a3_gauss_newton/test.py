import sys
sys.path.append("../..")
from solution import solve
from optalg.example_nlps.logistic import Logistic
from optalg.example_nlps.trajectory import PointTrajOpt
from optalg.example_nlps.f_r import F_R
from optalg.example_nlps.linear_least_squares import LinearLeastSquares
from optalg.interface.nlp_traced import NLPTraced
import unittest
import numpy as np

FACTOR = 20


class testSolver(unittest.TestCase):
    """
    """

    def testNonLinearSOS(self):
        problem = NLPTraced(Logistic(), FACTOR * 186)
        x = solve(problem)
        self.assertTrue(np.linalg.norm(problem.nlp.xopt - x) < 1e-3)

    def test_F_R(self):
        n = 10
        Q = np.diag(100. * np.arange(0, 10))
        R = .5 * np.eye(n)
        d = np.zeros(n)

        problem = NLPTraced(F_R(Q, R, d), FACTOR * 17)
        x = solve(problem)
        solution = np.zeros(n)
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testLinearLSEasy(self):

        A = np.array([[1., 2.], [3., 4.]])
        b = np.array([2., 4.])

        problem = NLPTraced(
            LinearLeastSquares(
                A, b, add_reg=False), 13 * FACTOR)

        x = solve(problem)
        solution = np.array([0., 1.])
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testLinearSOS(self):

        A = np.array([[0.46673604,
                       0.31534857,
                       0.80247336,
                       0.52129267,
                       0.13537983,
                       0.0450212,
                       0.74143284,
                       0.99978296,
                       0.94447352,
                       0.42050328],
                      [0.62190357,
                       0.62643564,
                       0.74369398,
                       0.44580501,
                       0.91381145,
                       0.75865574,
                       0.84341515,
                       0.31358747,
                       0.77024385,
                       0.20874065],
                      [0.16792137,
                       0.84952289,
                       0.25338644,
                       0.60532127,
                       0.05460647,
                       0.78873751,
                       0.30452665,
                       0.14602035,
                       0.53280784,
                       0.35635996],
                      [0.44939264,
                       0.26284333,
                       0.14598262,
                       0.41328202,
                       0.28844791,
                       0.82292302,
                       0.38184757,
                       0.40011535,
                       0.89619563,
                       0.15450804],
                      [0.50574735,
                       0.58081357,
                       0.52304145,
                       0.18120874,
                       0.2793881,
                       0.00421866,
                       0.12596743,
                       0.02581518,
                       0.19324229,
                       0.06150891],
                      [0.08829924,
                       0.14360407,
                       0.00727602,
                       0.21033869,
                       0.78615517,
                       0.9814616,
                       0.69569071,
                       0.5459732,
                       0.54713181,
                       0.77328764],
                      [0.28527904,
                       0.94937211,
                       0.2062728,
                       0.89899143,
                       0.72035624,
                       0.41148041,
                       0.69497496,
                       0.65577434,
                       0.58181849,
                       0.55580064],
                      [0.20051891,
                       0.57872251,
                       0.21725229,
                       0.39032888,
                       0.87235208,
                       0.16008489,
                       0.36873708,
                       0.5317815,
                       0.60444602,
                       0.01302496]])

        b = np.zeros(8)

        problem = NLPTraced(
            LinearLeastSquares(A, b), 13 * FACTOR)

        x = solve(problem)
        solution = np.zeros(10)
        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testTrajectory(self):
        N = 40
        problem = NLPTraced(PointTrajOpt(N), 29 * FACTOR)
        x = solve(problem)

        solution = np.array([0.07145219, 0.0047813, 0.14290438, 0.0095626, 0.21435657, 0.0143439,
                             0.28580877, 0.01912519, 0.35726096, 0.02390649, 0.42871315, 0.02868779,
                             0.50016534, 0.03346909, 0.57161753, 0.03825039, 0.64306972, 0.04303168,
                             0.71452191, 0.04781298, 0.7859741, 0.05259428, 0.8574263, 0.05737558,
                             0.92887849, 0.06215687, 1.00033068, 0.06693817, 1.07211503, 0.13895804,
                             1.14389939, 0.2109779, 1.21568374, 0.28299777, 1.28746809, 0.35501763,
                             1.35925245, 0.4270375, 1.4310368, 0.49905736, 1.50282116, 0.57107723,
                             1.57460551, 0.64309709, 1.64638987, 0.71511696, 1.71817422, 0.78713682,
                             1.78995858, 0.85915669, 1.86174293, 0.93117655, 1.93352729, 1.00319642,
                             1.9385441, 1.07842688, 1.9435609, 1.15365734, 1.94857771, 1.22888779,
                             1.95359452, 1.30411825, 1.95861133, 1.37934871, 1.96362814, 1.45457917,
                             1.96864495, 1.52980963, 1.97366175, 1.60504009, 1.97867856, 1.68027055,
                             1.98369537, 1.75550101, 1.98871218, 1.83073147, 1.99372899, 1.90596193,
                             1.9987458, 1.98119239])

        self.assertTrue(np.linalg.norm(x - solution) < 1e-3)


# Run tests with:
# python3 test.py
# Too see help and options
# python3 test.py --help
if __name__ == "__main__":
    unittest.main()
