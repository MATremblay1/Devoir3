import numpy as np
from numpy.linalg import solve, norm
import scipy.linalg as la

def newton(beta_init, F, J, tol, nmax):
    '''
    Args:
        beta_init : point initial (vecteur colonne 3x1)
        F         : fonction vectorielle (vecteur colonne nx1) dont on cherche le zero. DEPEND DE BETA.
        J         : matrice jacobienne de F (matrice nxn). DEPEND DE BETA.
        tol       : tolerance pour determiner la convergence
        nmax      : nombre maximal d'iterations

    Returns:
        beta      : vecteur des coefficients de la courbe de regression
    '''

    beta = beta_init.copy()
    assert beta.shape == (3,1), "beta_init doit être un vecteur colonne!"
    n = 0
    res = norm(F(beta.flatten()))
    dbeta = np.ones((3,1)) # Initialisation de dbeta pour entrer dans la boucle

    while res > tol and norm(dbeta) > tol and n < nmax:
        Jb = J(beta.flatten())
        Fb = F(beta.flatten())

        Q, R = la.qr(Jb, mode='economic')
        dbeta = solve(R, -Q.T @ Fb)

        # Mise à jour
        beta = beta + dbeta

        # Nouveau résidu
        res = norm(F(beta.flatten()))

        print(f"Iter {n+1} : ||dbeta|| = {norm(dbeta):.3e}, ||F(beta)|| = {res:.3e}")

        n += 1;

    return beta