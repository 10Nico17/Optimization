import sys
sys.path.append("/home/yuxiang/Documents/optalg22/optimization_algorithms_w22")
import numpy as np

from solution import solve
from optalg.example_nlps.logistic import Logistic
from optalg.example_nlps.trajectory import PointTrajOpt
from optalg.example_nlps.f_r import F_R
from optalg.example_nlps.linear_least_squares import LinearLeastSquares
from optalg.interface.nlp_traced import NLPTraced


problem = Logistic()
x = solve(problem)
print(x)
print(problem.xopt)
