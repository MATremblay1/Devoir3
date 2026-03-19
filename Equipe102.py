import numpy as np
import matplotlib.pyplot as plt
from time import time as time
from reglinA import reglinA
from reglinB import reglinB
from newton import newton


#a
# But: On cherche à représenter graphiquement les points du fichier points.txt
# afin de visualiser la tendance générale des données à l'aide du nuage de points. 
# On commence par créer la matrice X à partir du fichier points.txt
X = np.loadtxt("points.txt")
# Ensuite, on sépare les colonnes de X en x et y pour ensuite tracer le nuage de points
x = X[:, 0]
y = X[:, 1]

# Enfin, on trace le graphique du nuage de points
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure 1:\n Dispersion des points de y en fonction de x")
plt.show()
# Ainsi, on peut voir que les points semblent suivre une tendance plutôt linéaire.

#c
# But: On cherche la droite qui approxime le mieux les données

betaA = reglinA(X)
betaB = reglinB(X)

print("Beta approche A =", betaA)
print("Beta approche B =", betaB)

xx = np.linspace(1, 6)

yA = betaA[0] + betaA[1]*xx
yB = betaB[0] + betaB[1]*xx

plt.figure()
plt.scatter(x, y, s=3, label="Données")
plt.plot(xx, yA, label="Régression A", color="red", linewidth=3)
plt.plot(xx, yB, label="Régression B", color="lime", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure 2:\n Régression linéaire par moindres carrés (approches A et B)")
plt.legend()
plt.show()


# f)
def F(beta):
    b1, b2, b3 = beta
    return (b1 + b2*np.sqrt(x - b3) - y).reshape(-1, 1)

def J(beta):
    b1, b2, b3 = beta
    n = len(x)

    J = np.zeros((n, 3))
    J[:, 0] = 1
    J[:, 1] = np.sqrt(x - b3)
    J[:, 2] = -b2 / (2*np.sqrt(x - b3))
    return J

beta0 = np.array([[1.0], [1.0], [1.0]])

beta_newton = newton(beta0, F, J, tol=1e-7, nmax=20)
print("\nBeta Newton =")
print(beta_newton)

xx = np.linspace(1, 6)
yy = beta_newton[0] + beta_newton[1]*np.sqrt(xx - beta_newton[2])

plt.figure()
plt.scatter(x, y, s=3, label="Données")
plt.plot(xx, yy, label="Régression non-linéaire", color='orange', linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure 3:\n Régression non-linéaire de f(x) par Newton modifié")
plt.legend()
plt.show()
#Test 
#Je refais une tentative de commit, faudrait que ça commit
# TESTY TEST TEST TEST TEST