import sys
sys.path.append("../..")


from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
import numpy as np

class NLP_Gaussina_ineq(NLP):
    """
    Nonlinear program

    Cost:  - exp ( - (x - x0)^T D (x - x0))
    Inequalities: Ax <= b
    Variable: x in R^n

    Parameters:
    x0 in R^n
    D in R^(nxn) symmetric
    A in R^(mxn)
    b in R^m

    ^T means transpose
    exp is exponential function

    Feature types: [ OT.f ] +  m * [ OT.ineq ]

    """

    def __init__(self, x0, D, A, b):
        """
        """
        self.x0 = x0
        self.D = D
        self.A = A
        self.b = b

    def evaluate(self, x):
        """
        Returns the features (y) and the Jacobian (J) of the nonlinear program.

        In this case, we have 1 cost function and m inequalities.
        The cost should be in the first entry (index 0) of the feature
        vector. The inequality features should come next, following the
        natural order in Ax<=b. That is, the first inequality (second entry of
        the feature vector) is A[0,:] x <= b[0], the second inequality
        is A[1,:] x <= b[1] and so on.

        The inequality entries should be written in the form y[i] <= 0.
        For example, for inequality x[0] <= 1 --> we use feature
        y[i] = x[0] - 1.

        The row i of the Jacobian J is the gradient of the entry i in
        the feature vector, e.g. J[0,:] is the gradient of y[0].

        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1+m,))
            J: the Jacobian (2-D np.ndarray of shape (1+m,n))

        See also:
        ----
        NLP.evaluate
        """
        #Cost:  - exp ( - (x - x0)^T D (x - x0))
        e = np.e
        y = np.zeros(np.shape(self.A)[0] + 1)
        J = np.zeros((np.shape(self.A)[0] + 1, NLP_Gaussina_ineq.getDimension(self)))
        y0 = -e**(-(x-self.x0)@self.D@(x-self.x0).T)
        J0 = -y0*(self.D+self.D.T)@(x-self.x0)
        for ind, val in enumerate(y):
            if ind == 0:
                y[ind] = y0
                J[ind,:] = J0
            else:
                y[ind] = (self.A[ind-1,:]@x) - self.b[ind - 1]
                J[ind] = self.A[ind-1,:]
        

        return y, J


    def getDimension(self):
        """
        Return the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """

        n = np.shape(self.x0)[0]
        return n


    def getFHessian(self, x):
        """
        Returns the hessian of the cost term.
        The output should be:
            H: the hessian (2-D np.ndarray of shape (n,n))

        See Also
        ------
        NLP.getFHessian
        """
        d = np.shape(x)[0]
        I = np.ones(d)
        fx = NLP_Gaussina_ineq.evaluate(self, x)[0][0]
        DD = self.D + np.transpose(self.D)
        xx = x-self.x0
        pp1 = (DD@xx).reshape((-1,1))
        pp2 = (xx@DD).reshape(1,-1)
        pp = pp1@pp2
        H = -fx*(DD-pp)
        return H


    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return [OT.f] + self.A.shape[0] * [OT.ineq]

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        return "Gaussian function with inequalities"
