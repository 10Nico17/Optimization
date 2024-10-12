import numpy as np

#M = np.array(([2, -1, 0], [-1, 2, -1], [0, -1, 2]), dtype=float)
M = np.array(([-2, 4], [4, -8]), dtype=float)
#M = np.array(([-2, 2], [2, -4]), dtype=float)

eigs = np.linalg.eigvals(M)
print("The eigenvalues of M:", eigs)

if (np.all(eigs>0)):
    print("M is positive definite")
elif (np.all(eigs>=0)):
    print("M is positive semi-definite")
else:
    print("M is negative definite")