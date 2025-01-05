import sys
sys.path.append("../..")
from optalg.interface.nlp import NLP
import numpy as np


class NLP_nonlinear(NLP):
    """
    Nonlinear program with cost  1 / || C x ||^2
    x in R^n
    C in R^(m x n)
    || . || is the 2-norm
    feature types: [ OT.f ]

    """

    def __init__(self, C):
        """
        """
        self.C = C

    def evaluate(self, x):
        """
        Returns the features and the Jacobians
        of a nonlinear program.
        In this case, we have a single feature (the cost function)
        because there are no constraints or residual terms.
        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1,)) 
            J: the jacobian (2-D np.ndarray of shape (1,n))

        See also:
        ----
        NLP.evaluate
        """
        C_0 = self.C
        dm_1 = np.shape(C_0)[0]
        dm_2 = np.shape(C_0)[1]
        dm_x = np.shape(x)[0]
        y = 1 / (np.linalg.norm(C_0@x, 2))**2
        J = np.zeros(dm_x)
        def det_solve(pos:int):
            result = 0
            for i in range(dm_1):
                for j in range(dm_2):
                    temp = 2*C_0[i,pos]*C_0[i,j]*x[j]
                    result += temp
            return result
        for i in range(dm_2):
            J[i] = (-(y**2))*det_solve(i)
        return  np.array([y]) , np.array([J])

    def getDimension(self):
        """
        Returns the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """

        # n =
        # return n

    def getFHessian(self, x):
        """
        Returns the hessian of the cost term.
        The output should be: 
            H: the hessian (2-D np.ndarray of shape (n,n))

        See Also
        ------
        NLP.getFHessian
        """
        # add code to compute the Hessian matrix
        C_0 = self.C
        dm_x = np.shape(x)[0]
        C_t = np.transpose(C_0)
        num_n = np.shape(C_0)[1]
        num_m = np.shape(C_0)[0]
        y,J = NLP_nonlinear.evaluate(self,x)
        def J2(pos_i, pos_j):
            result = 0
            for i in range(num_m):
                    temp = 2*(C_0[i, pos_i]*C_0[i, pos_j])
                    result += temp
            return result
        dif = np.zeros((dm_x, dm_x))
        for i in range(dm_x):
            for j in range(dm_x):
                dif[i, j] = 2*(y[0]**3)*((J[0][i]/((-(y[0]**2))))*J[0][j]/((-(y[0]**2))))-(y[0]**2)*(J2(i, j))
        H = dif
        
        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        return "Nonlinear function  1 / || C x ||^2"
