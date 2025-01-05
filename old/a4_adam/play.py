import sys
sys.path.append("/home/yuxiang/Documents/optalg22/optimization_algorithms_w22")
import numpy as np
from optalg.example_nlps.quadratic_stochastic import Quadratic_stochastic
from optalg.example_nlps.linear_least_squares_stochastic import Linear_least_squares_stochastic
from solution import solve
test = 3
match test:
    case 1:
        A = np.array([[0.85880983, 0.42075752, 0.14625862],
                    [0.57705246, 0.76635021, 0.64077446],
                    [0.85916002, 0.86186594, 0.81231712],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]])

        b = np.zeros(6)

        problem = Linear_least_squares_stochastic(A, b)
        x_init = problem.getInitializationSample()
        x, list, t = solve(problem)

    case 2:
        A = np.array([[0.90657231, 0.59956825, 0.79181794],
                      [0.28089784, 0.27284522, 0.6512166],
                      [0.21775311, 0.81865644, 0.95784162],
                      [0.47410968, 0.98784787, 0.5146119],
                      [0.83881493, 0.60040309, 0.19827599],
                      [0.90834492, 0.66617969, 0.83357351],
                      [0.97028032, 0.67150341, 0.58111735],
                      [0.55876076, 0.36616212, 0.87166266],
                      [0.7202108, 0.85757553, 0.41819739],
                      [0.54088318, 0.46910575, 0.45215856]])

        b = np.array([0.08833689,
                      0.07526668,
                      0.01135477,
                      0.06107546,
                      0.05448763,
                      0.08905451,
                      0.06238092,
                      0.09981542,
                      0.0489913,
                      0.02405487])

        problem = Linear_least_squares_stochastic(A, b)
        x_init = problem.getInitializationSample()
        x, list, t = solve(problem)
    
    case 3:
        Qs = 5 * [np.diag([100., 1., 0.1])]

        gs = [np.array([0.13359639,
                        0.20692249,
                        0.4099303]),
              np.array([0.86686032,
                        0.26361185,
                        0.48886114]),
              np.array([0.01843443,
                        0.89926551,
                        0.3261251]),
              np.array([0.86751661,
                        0.95695649,
                        0.68315632]),
              np.array([0.90034537,
                        0.96245117,
                        0.84465686])]

        problem = Quadratic_stochastic(Qs, gs)
        x, list, t = solve(problem)
        x_init = problem.getInitializationSample()
print("init:",x_init, "\noutput", x,"\nlist:", list, "\nt:", t)
print("otpimal solution", problem.opt)
