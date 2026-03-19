import numpy as np
from scipy.linalg import qr
from numpy.linalg import solve, norm

def regfreqB(X, k) :
    '''
    Args:
        X : le jeu de donnees n x 2
        k : nombre de fréquences

    Returns :
        beta : vecteur des coefficients de la régression de Fourier
    '''
    n = X.shape[0] # Nombre de points

    # Creation du membre de droite
    y = X[:,1].reshape(n,1)

    # Creation de la matrice A
    x = X[:,0].reshape(n,1)
    cosinus = np.hstack([np.cos(j*x) for j in range(1,k)])
    sinus = np.hstack([np.sin(j*x) for j in range(1,k)])
    A = np.hstack((np.ones((n,1)), cosinus, sinus))

    # Resolution du systeme rectangulaire (approche B)
    Q, R = qr(A, mode='economic')
    beta = solve(R,Q.T@y)

    print(f"Norme du residu ||F(beta)|| = ||A*beta - y|| = {norm(A@beta-y)}")

    return beta