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
        """
        """
        q1 = x[0]
        q2 = x[1]
        q3 = x[2]
        q4 = x[3]
        pm1 = self.pr[0]
        pm2 = self.pr[1]
        q01 = self.q0[0]
        q02 = self.q0[1]
        q03 = self.q0[2]
        q04 = self.q0[3]
        lamda = self.l
        sl = np.sqrt(lamda)
        # f1 = ((np.cos(q1 + q2) / 2)- pm1 + np.cos(q1) + np.cos(q1+q2+q3)*(q4 + (1/3)))**2
        # f2 = ((np.sin(q1 + q2) / 2)- pm2 + np.sin(q1) + np.sin(q1+q2+q3)*(q4 + (1/3)))**2
        # f3 = (q1-q01)**2 + (q2-q02)**2 + (q3-q03)**2 + (q4-q04)**2
        # f01 = (q1-q01)**2
        # f02 = (q2-q02)**2 
        # f03 = (q3-q03)**2
        # f04 = (q4-q04)**2
        # y = f1 + f2 + f3
        # uni_term_f1 = ((np.cos(q1+q2)/2) -  pm1 + np.cos(q1) + np.cos(q1+q2+q3)*(q4+1/3))
        # uni_term_f2 = ((np.sin(q1+q2)/2) -  pm2 + np.sin(q1) + np.sin(q1+q2+q3)*(q4+1/3))
        # df1dq1 = -2*((np.sin(q1+q2)/2) + np.sin(q1) + np.sin(q1+q2+q3)*(q4+1/3))*uni_term_f1
        # df1dq2 = -2*((np.sin(q1+q2)/2) + np.sin(q1+q2+q3)*(q4+1/3))*uni_term_f1
        # df1dq3 = -2*np.sin(q1+q2+q3)*(q4+1/3)*uni_term_f1
        # df1dq4 = 2*np.cos(q1+q2+q3)*uni_term_f1
        # df2dq1 = 2*(uni_term_f1 + pm1)*uni_term_f2
        # df2dq2 = 2*(uni_term_f1 + pm1 - np.cos(q1))*uni_term_f2
        # df2dq3 = 2*np.cos(q1+q2+q3)*(q4+1/3)*uni_term_f2
        # df2dq4 = 2*np.sin(q1+q2+q3)*uni_term_f2
        # df3dq1 = 2*self.l*(q1 - q01)
        # df3dq2 = 2*self.l*(q2 - q02)
        # df3dq3 = 2*self.l*(q3 - q03)
        # df3dq4 = 2*self.l*(q4 - q04)
        # J = np.array([df1dq1+df2dq1+df3dq1,
        #             df1dq2+df2dq2+df3dq2,
        #             df1dq3+df2dq3+df3dq3,
        #             df1dq4+df2dq4+df3dq4])
        # r1 = f1
        # r2 = f2
        # r3 = f01
        # r4 = f02
        # r5 = f03
        # r6 = f04
        # r = np.array([r1, r2, r3, r4, r5, r6])
        # y_r = np.sum(r)
        # J_r_1 = [df1dq1, df1dq2, df1dq3, df1dq4]
        # J_r_2 = [df2dq1, df2dq2, df2dq3, df2dq4]
        # J_r_3 = [df3dq1,0,0,0]
        # J_r_4 = [0,df3dq2,0,0]
        # J_r_5 = [0,0,df3dq3, 0]
        # J_r_6 = [0,0,0,df3dq4]
        # J_r = np.array([J_r_1, J_r_2, J_r_3, J_r_4, J_r_5, J_r_6])
        # J_r = np.transpose(J_r)
        p1 = np.cos(q1) + 0.5*np.cos(q1+q2) + ((1/3) + q4)*np.cos(q1+q2+q3)
        p2 = np.sin(q1) + 0.5*np.sin(q1+q2) + ((1/3) + q4)*np.sin(q1+q2+q3)
        r1 = p1-pm1
        r2 = p2-pm2
        r3 = (q1-q01)*sl
        r4 = (q2-q02)*sl
        r5 = (q3-q03)*sl
        r6 = (q4-q04)*sl
        uni_term_p1 = ((1/3)+q4)*np.sin(q1+q2+q3)
        uni_term_p2 = ((1/3)+q4)*np.cos(q1+q2+q3)
        dp1dq1 = -np.sin(q1) - 0.5*np.sin(q1+q2) - uni_term_p1
        dp1dq2 = -0.5*np.sin(q1+q2) - uni_term_p1
        dp1dq3 = -uni_term_p1
        dp1dq4 = np.cos(q1+q2+q3)
        dp2dq1 = np.cos(q1) + 0.5*np.cos(q1+q2) + uni_term_p2
        dp2dq2 = 0.5*np.cos(q1+q2) + uni_term_p2
        dp2dq3 = uni_term_p2
        dp2dq4 = np.sin(q1+q2+q3)
        dr1 = [dp1dq1, dp1dq2, dp1dq3, dp1dq4]
        dr2 = [dp2dq1, dp2dq2, dp2dq3, dp2dq4]
        dr3 = [sl,0,0,0]
        dr4 = [0,sl,0,0]
        dr5 = [0,0,sl,0]
        dr6 = [0,0,0,sl]
        r = np.array([r1, r2, r3, r4, r5, r6])
        dr = np.array([dr1, dr2, dr3, dr4, dr5, dr6])



        return r,dr

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
