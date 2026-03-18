import numpy as np
from scipy.linalg import qr
from numpy.linalg import solve, norm

def reglinB(X) :
    '''
    Args:
        X : le jeu de donnees n x 2

    Returns :
        beta : vecteur des coefficients de la droite de regression
    '''
    n = X.shape[0] # Nombre de points

    # Creation du membre de droite
    y = X[:,1].reshape(n,1)

    # Creation de la matrice A
    A = np.hstack((np.ones((n,1)), X[:,0].reshape((n,1))))

    # Resolution du systeme rectangulaire (approche B)
    Q, R = qr(A, mode='economic')
    beta = solve(R,Q.T@y)

    print(f"Norme du residu ||F(beta)|| = ||A*beta - y|| = {norm(A@beta-y)}")

    return beta