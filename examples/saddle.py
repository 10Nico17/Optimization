import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definiere die Sattelpunktsfunktion f(x1, x2) = x1^2 - x2^2
def saddle_function(x1, x2):
    return x1**2 - x2**2

# Erstelle ein Gitter für die x1- und x2-Werte
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = saddle_function(X1, X2)

# Erstelle die 3D-Plot-Visualisierung
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Beschriftungen und Titel
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Sattelpunktsfunktion f(x1, x2) = x1^2 - x2^2')

plt.show()


# Erstelle den Konturplot
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=20, cmap='coolwarm')
plt.clabel(contours, inline=True, fontsize=8)

# Markiere den Sattelpunkt (0, 0)
plt.scatter(0, 0, color='red', s=100, label='Saddle Point (0, 0)')

# Beschriftungen und Titel
plt.title(r'Saddle point (0, 0) on the saddle surface $f(\mathbf{x}) = x_1^2 - x_2^2$')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()