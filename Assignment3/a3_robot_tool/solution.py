import numpy as np
import sys
sys.path.append("../..")
from optalg.interface.nlp import NLP
import math


class RobotTool(NLP):
    """
    """

    def __init__(self, q0: np.ndarray, pr: np.ndarray, l: float):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        self.q0 = q0
        self.pr = pr
        self.l = l


    def evaluate(self, x):
        theta1 = x[0]
        theta2 = x[1]
        theta3 = x[2]
        theta4 = x[3]
        target_x = self.pr[0]
        target_y = self.pr[1]
        init_theta1 = self.q0[0]
        init_theta2 = self.q0[1]
        init_theta3 = self.q0[2]
        init_theta4 = self.q0[3]
        weight = self.l
        sqrt_w = np.sqrt(weight)

        pos_x = np.cos(theta1) + 0.5 * np.cos(theta1 + theta2) + ((1/3) + theta4) * np.cos(theta1 + theta2 + theta3)
        pos_y = np.sin(theta1) + 0.5 * np.sin(theta1 + theta2) + ((1/3) + theta4) * np.sin(theta1 + theta2 + theta3)

        res_x = pos_x - target_x
        res_y = pos_y - target_y
        res_theta1 = (theta1 - init_theta1) * sqrt_w
        res_theta2 = (theta2 - init_theta2) * sqrt_w
        res_theta3 = (theta3 - init_theta3) * sqrt_w
        res_theta4 = (theta4 - init_theta4) * sqrt_w

        jac_term_x = ((1/3) + theta4) * np.sin(theta1 + theta2 + theta3)
        jac_term_y = ((1/3) + theta4) * np.cos(theta1 + theta2 + theta3)

        dpos_x_dtheta1 = -np.sin(theta1) - 0.5 * np.sin(theta1 + theta2) - jac_term_x
        dpos_x_dtheta2 = -0.5 * np.sin(theta1 + theta2) - jac_term_x
        dpos_x_dtheta3 = -jac_term_x
        dpos_x_dtheta4 = np.cos(theta1 + theta2 + theta3)

        dpos_y_dtheta1 = np.cos(theta1) + 0.5 * np.cos(theta1 + theta2) + jac_term_y
        dpos_y_dtheta2 = 0.5 * np.cos(theta1 + theta2) + jac_term_y
        dpos_y_dtheta3 = jac_term_y
        dpos_y_dtheta4 = np.sin(theta1 + theta2 + theta3)

        dres_x = [dpos_x_dtheta1, dpos_x_dtheta2, dpos_x_dtheta3, dpos_x_dtheta4]
        dres_y = [dpos_y_dtheta1, dpos_y_dtheta2, dpos_y_dtheta3, dpos_y_dtheta4]
        dres_theta1 = [sqrt_w, 0, 0, 0]
        dres_theta2 = [0, sqrt_w, 0, 0]
        dres_theta3 = [0, 0, sqrt_w, 0]
        dres_theta4 = [0, 0, 0, sqrt_w]

        y = np.array([res_x, res_y, res_theta1, res_theta2, res_theta3, res_theta4])
        J = np.array([dres_x, dres_y, dres_theta1, dres_theta2, dres_theta3, dres_theta4])

        return y, J




    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 4

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.q0
