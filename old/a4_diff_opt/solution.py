import numpy as np
import sys
sys.path.append("../..")
from optalg.interface.nlp import NLP
import math


QP_0 = {
    "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    "b": np.array([1., 1., 1., 1.]),
    "x": np.array([.5, .5]),
    "yopt": np.array([.5, .5]),
    "lopt": np.array([0., 0., 0., 0.])
}


QP_1 = {
    "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    "b": np.array([1., 1., 1., 1.]),
    "x": np.array([2, 0]),
    "yopt": np.array([1, 0]),
    "lopt": np.array([2, 0., 0., 0.])
}

QP_2 = {
    "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    "b": np.array([1., 1., 1., 1.]),
    "x": np.array([2, 2]),
    "yopt": np.array([1, 1]),
    "lopt": np.array([2., 2., 0, 0]),


}


QPs = [QP_0, QP_1, QP_2]


def oracle(x: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Oracle that provides solution to QP
    min_y || y - x ||^2
    s.t. Ay <= b

    Arguments:
    ---
    x: 1-D np.array
    A: 2-D np.array
    b: 1-D np.array


    Returns:
    yopt: 1-D np.array
    lopt: 1-D np.array

    """

    found = False
    num_qps = len(QPs)
    id = 0
    while (id < num_qps and not found):
        QP = QPs[id]
        if np.allclose(
                QP["A"],
                A) and np.allclose(
                QP["b"],
                b) and np.allclose(
                QP["x"],
                x):
            found = True
            yopt = QP["yopt"]
            lopt = QP["lopt"]
            # Check that the solution of the oracle is correct.
            assert np.sum(QP["A"] @ yopt - QP["b"] > 0) == 0
            assert np.sum(lopt < 0) == 0
            assert np.sum(lopt * (QP["A"] @ yopt - QP["b"]) != 0) == 0
            assert np.linalg.norm(
                2 * (yopt - QP["x"]) + lopt @ QP["A"]) < 1e-12
        id += 1
    if not found:
        raise RuntimeError("QP is not in the database")

    return yopt, lopt


class DiffOpt(NLP):
    """
    min_x c^T yopt(x)

    where yopt(x) is the solution of min_y || y - x ||^2 s.t. Ay <= b

    x in R^2, y in R^2,  A in R^(2x2), and b in R^2
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        """
        Arguments
        ----
        c: 1-D np.array
        A: 2-D np.array
        b: 1-D
        """
        self.c = c
        self.A = A
        self.b = b

    def evaluate(self, x):
        """
        Returns the features (y) and the Jacobian (J) of the nonlinear program.
        In this case, we have 1 cost function.

        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1,))
            J: the Jacobian (2-D np.ndarray of shape (1,n))


        Notes:
        ---

        For an input x, you can get the optimal yopt calling the oracle:

        yopt, lopt = oracle(x, self.A, self.b)

        yopt is the optimal y.
        lopt is the value of the Lagrange multipliers.

        The Jacobian dyopt/dx has to be computed using the implicit function theorem on the KKT conditions
        of the optimization problem,

        min_y || y - x ||^2 s.t. Ay <= b

        where y is the variable and x is the parameter.

        """
        yopt, lopt = oracle(x, self.A, self.b)
        x1 = x[0]
        x2 = x[1]
        y1 = yopt[0]
        y2 = yopt[1]
        l1 = lopt[0]
        l2 = lopt[1]
        l3 = lopt[2]
        l4 = lopt[3]
        g = (self.A@yopt - self.b)
        A_1 = np.transpose(self.A)[0]
        A_2 = np.transpose(self.A)[1]
        l_A_1 = lopt * A_1
        l_A_2 = lopt * A_2
        sum_l_A_1 = np.sum(l_A_1)
        sum_l_A_2 = np.sum(l_A_2)
        dfdy1 = 2*(y1 - x1) + sum_l_A_1
        dfdy2 = 2*(y2 - x2) + sum_l_A_2
        df = np.array([dfdy1, dfdy2])
        diag_l = np.diag(lopt)
        l_g = diag_l@g
        r = np.concatenate((df,l_g),axis = None)
        dfdy1y1 = 2
        dfdy1y2 = 0
        dfdy2y1 = 0
        dfdy2y2 = 2
        dg = self.A
        diag_g = np.diag(g)
        drdyy = np.array([[dfdy1y1, dfdy1y2],
                            [dfdy2y1, dfdy2y2]])
        dldg = diag_l@dg
        dg = np.transpose(self.A)
        drdyl_1 = np.concatenate((drdyy[0],A_1), axis = None)
        drdyl_2 = np.concatenate((drdyy[1],A_2), axis = None)
        drdyl_3 = np.concatenate((dldg[0],diag_g[0]), axis = None)
        drdyl_4 = np.concatenate((dldg[1],diag_g[1]), axis = None)
        drdyl_5 = np.concatenate((dldg[2],diag_g[2]), axis = None)
        drdyl_6 = np.concatenate((dldg[3],diag_g[3]), axis = None)
        drdyl = np.array([drdyl_1,
                            drdyl_2,
                            drdyl_3,
                            drdyl_4,
                            drdyl_5,
                            drdyl_6])
        dgdx = np.zeros((4,2))
        dfdyx = np.array([[-2,0],
                            [0,-2]])
        drdx = np.concatenate((dfdyx,dgdx), axis = 0)
        inv_drdyl = np.linalg.inv(drdyl)
        dydx = -inv_drdyl@drdx
        tdydx = dydx[0:2]
        y = self.c.T@yopt
        J = self.c@tdydx
        # y = ...
        # J = ...


        #return dfdx, dfdy
        #return yopt, lopt, x
        #return x1, x2
        #return y1, y2
        #return l_A_y
        return y,J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 2
