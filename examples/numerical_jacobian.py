import numpy as np

def finite_difference_jacobian_check(x, f, df, epsilon=1e-6):
    n = x.shape[0]
    d = f(x).shape[0]
    
    # Initialize the Jacobian approximation
    J_hat = np.zeros((d, n))
    
    # Loop through each dimension
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        # Compute the ith column of the Jacobian approximation
        J_hat[:, i] = (f(x + epsilon * e_i) - f(x - epsilon * e_i)) / (2 * epsilon)
    
    # Check if the difference between the approximate and true Jacobian is small
    if np.linalg.norm(J_hat - df(x), ord=np.inf) < 1e-4:
        return True
    else:
        return False

# Definiere die Funktion f(x)
def f(x):
    return np.array([np.sin(x[0]), x[0]**2])

# Definiere die exakte Jacobian-Funktion df(x)
def df(x):
    return np.array([[np.cos(x[0])], [2*x[0]]])

# Wähle einen Punkt x, bei dem du die Überprüfung durchführen möchtest
x = np.array([1.0])

# Rufe die finite_difference_jacobian_check Funktion auf
result = finite_difference_jacobian_check(x, f, df)

print("Jacobian check passed:", result)