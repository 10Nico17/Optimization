import numpy as np
import matplotlib.pyplot as plt

# Definiere die Funktion f(x1, x2) = x1^2 + x2^2
def func(x1, x2):
    return x1**2 + x2**2

# Definiere die Ableitungen (Gradient) der Funktion
def gradient(x1, x2):
    grad_x1 = 2 * x1
    grad_x2 = 2 * x2
    return grad_x1, grad_x2

# Erstelle ein Gitter für die x1- und x2-Werte
x1 = np.linspace(-20, 20, 20)
x2 = np.linspace(-20, 20, 20)
X1, X2 = np.meshgrid(x1, x2)
Z = func(X1, X2)

# Berechne den Gradienten auf dem Gitter
grad_X1, grad_X2 = gradient(X1, X2)

# Erstelle den Konturplot
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=20, cmap='coolwarm')
plt.clabel(contours, inline=True, fontsize=8)

# Erstelle den Vektorplot (Gradienten)
plt.quiver(X1, X2, grad_X1, grad_X2, color='gray')

# Beschriftungen und Titel
plt.title(r'Gradient Field and Contours of $f(\mathbf{x}) = x_1^2 + x_2^2$')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()
