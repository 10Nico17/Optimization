import sys
import numpy as np

sys.path.append("../..")
from optalg.utils.finite_diff import *
from solution import RobotTool

q0 = np.zeros(4)
pr = np.array([.5, 2.0 / 3.0])
l = .5
problem = RobotTool(q0, pr, l)
x = np.pi / 180.0 * np.array([90, -90, -90, 0.])
phi, J = problem.evaluate(x)
print(phi)
print(J)