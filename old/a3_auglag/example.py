import sys
sys.path.append("../..")
import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.utils.finite_diff import *
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plotFunc(f, bounds_lo, bounds_up, trace_xy=None, title=None):
    x = np.linspace(bounds_lo[0], bounds_up[0], 100)
    y = np.linspace(bounds_lo[1], bounds_up[1], 100)
    xMesh, yMesh = np.meshgrid(x, y, indexing='ij')
    zMesh = np.zeros_like(xMesh)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            zMesh[i, j] = f([xMesh[i, j], yMesh[i, j]])

    surf2 = plt.contourf(xMesh, yMesh, zMesh, cmap=cm.coolwarm)
    if trace_xy is not None:
        plt.plot(trace_xy[:, 0], trace_xy[:, 1], 'ko-')
    plt.colorbar(surf2)
    plt.xlabel('x')
    plt.ylabel('y')

    if title is not None:
        plt.title(title)

    plt.show()


class Cost_with_penalty(NLP):
    """
    NOTE: cost term and inequalities
    """

    def __init__(self, nlp):
        self.mu = 1.
        self.nlp = nlp
        types = self.nlp.getFeatureTypes()
        self.id_f = [i for i, t in enumerate(types) if t == OT.f]
        self.id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
        assert len([i for i, t in enumerate(types) if t == OT.r]) == 0
        assert len([i for i, t in enumerate(types) if t == OT.eq]) == 0
        self.x0 = self.nlp.getInitializationSample()

    def getInitializationSample(self):
        return self.x0

    def evaluate(self, x):

        y, J = self.nlp.evaluate(x)

        cost = 0
        grad = np.zeros(len(x))

        cost = y[self.id_f[0]]
        ineq = y[self.id_ineq]

        grad = np.copy(J[self.id_f[0]])
        Jineq = J[self.id_ineq]

        for i in range(len(self.id_ineq)):
            if ineq[i] >= 0:
                cost += self.mu * ineq[i] * ineq[i]
                grad += 2 * self.mu * ineq[i] * Jineq[i]

        # only numpy
        # ineq_active = ineq >= 0
        # cost += np.sum(self.mu * ineq_active * ineq ** 2)
        # grad += 2 * self.mu * Jineq.T @ (ineq_active * ineq)

        return np.array([cost]), grad.reshape(1, -1)

    def getFHessian(self, x):

        y, J = self.nlp.evaluate(x)

        H = self.nlp.getFHessian(x).copy()

        ineq = y[self.id_ineq]

        Jineq = J[self.id_ineq]

        for i in range(len(self.id_ineq)):
            if ineq[i] >= 0:
                H += 2 * self.mu * np.outer(Jineq[i], Jineq[i])

        # only numpy
        # ineq_active = ineq >= 0
        # H += self.mu * 2 * (Jineq.T * ineq_active) @  Jineq

        return H


def _solve(nlp, trace=None):
    """
        Simple Solver.
        Regularized newton steps with a fixed step size.
    """

    x = nlp.getInitializationSample()

    max_it = 30
    delta = 1e-4
    alpha = .1
    hessian_reg = 1
    n = len(x)

    xprev = np.copy(x)
    phi, J = nlp.evaluate(x)
    print(f"inner it: -1 x: {x} f:{phi[0]}")

    if trace is not None:
        trace.append(np.copy(x))
    for i in range(max_it):
        phi, J = nlp.evaluate(x)
        H = nlp.getFHessian(x)
        D = -np.linalg.solve(H + hessian_reg * np.eye(n), J[0])
        x = x + alpha * D

        phi, _ = nlp.evaluate(x)
        print(f"inner it: {i} x: {x} f:{phi[0]}")

        if np.linalg.norm(x - xprev) < delta:
            print("break inner xtol")
            break

        if trace is not None:
            trace.append(np.copy(x))
        xprev = np.copy(x)

    return x


def solve(nlp):
    """
    Solver for Constrained optimization (only cost term and inequalities)
    Squared Penalty Method
    """

    types = nlp.getFeatureTypes()
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    max_it = 20
    mu = 1.
    it = 0
    xtol = 1e-4
    mu_rate = 2

    x = nlp.getInitializationSample()
    xprev = x.copy()

    nlp_penalty = Cost_with_penalty(nlp)
    nlp_penalty.mu = mu

    print(f"outer initializatoin x {x}")
    while it < max_it:

        nlp_penalty.x0 = np.copy(xprev)
        nlp_penalty.mu = mu

        trace = []
        x = _solve(nlp_penalty, trace)
        trace_np = np.vstack(trace)

        def cost_penalty(x):
            y, J = nlp_penalty.evaluate(x)
            upper_bound_for_visualization = 5
            if y[0] > upper_bound_for_visualization:
                return upper_bound_for_visualization
            return y[0]

        print(
            f"Plotting Penalty function mu{nlp_penalty.mu} with optimizer trace")

        plotFunc(cost_penalty, [-2, -2], [2, 2],
                 trace_np, title=f"mu {nlp_penalty.mu}")

        y, _ = nlp.evaluate(x)
        print(f"outer it: {it} x: {x} f: {y[0]} gs: {y[id_ineq]}")

        mu *= mu_rate

        if (np.linalg.norm(x - xprev) < xtol):
            break

        xprev = x.copy()
        it += 1


# problem = HalfCircle()
# nlp_penalty = Cost_with_penalty(problem)

# print("Checking the Gradient and Hessian of Cost with Penalty")

# x = np.array([.1, .5])
# y, J = nlp_penalty.evaluate(x)
# H = nlp_penalty.getFHessian(x)
# eps = 1e-4
# Jdiff = finite_diff_J(nlp_penalty, x, eps)
# assert np.allclose(J, Jdiff, atol=10 * eps)


# def f_nlp_penalty(x):
#     return nlp_penalty.evaluate(x)[0][0]


# Hdiff = finite_diff_hess(f_nlp_penalty, x, eps)
# assert np.allclose(H, Hdiff, atol=10 * eps)

# x = np.array([-.1, .5])
# y, J = nlp_penalty.evaluate(x)
# H = nlp_penalty.getFHessian(x)
# eps = 1e-4
# Jdiff = finite_diff_J(nlp_penalty, x, eps)
# assert np.allclose(J, Jdiff, atol=10 * eps)


# Hdiff = finite_diff_hess(f_nlp_penalty, x, eps)
# assert np.allclose(H, Hdiff, atol=10 * eps)

problem = LinearProgramIneq
nlp_penalty = Cost_with_penalty(problem)
x = np.array([1, 1])
y, J = nlp_penalty.evaluate(x)
H = nlp_penalty.getFHessian(x)
eps = 1e-4
Jdiff = finite_diff_J(nlp_penalty, x, eps)
assert np.allclose(J, Jdiff, atol=10 * eps)


Hdiff = finite_diff_hess(f_nlp_penalty, x, eps)

# NOTE: Cost with Penalty use a Gauss Newton Approximation of
# the hessian of the constraints. If we evaluate in a point where
# a nonlinear constraint is active, the Hessian will be (slightly) different.

# assert np.allclose(H, Hdiff, atol=10 * eps)


print("plotting the original cost function")


def only_cost(x):
    return problem.evaluate(x)[0][0]


plotFunc(only_cost, [-2, -2], [2, 2], title="cost function")
plt.show()

print("plotting the cost function with penalty")


def cost_with_penalty(x):
    y, _ = nlp_penalty.evaluate(x)
    upper_bound_for_visualization = 5
    if y[0] > upper_bound_for_visualization:
        return upper_bound_for_visualization
    return y[0]


plotFunc(cost_with_penalty, [-2, -2], [2, 2], title="cost with penalty")
plt.show()


solve(problem)
