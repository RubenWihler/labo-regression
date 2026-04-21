from dual import DualNumber, np

### j = coordinate, f = fonction
def prepare_for_gradient( V , j ):
    """
    Prend le vecteur (v_0,v_1,...,v_n) et retourne (v_0,...,v_{j-1}, v_j + epsilon , v_{j+1},...,v_n)
    
    Paramètres
    ----------
    V : numpy array( num )
        num: type numérique
    j : int 
        index, 0 <= j < len(V)
              
    Retour
    ------
    numpy array( DualNumber )
    
    """
    ret = np.array([ DualNumber( k , 0 ) for k in V ])
    ret[j] = DualNumber( V[j] , 1 )
    return ret
    
def compute_gradient( f , P ):
    """
    Retourne la valeur du gradient de la fonction f au point V
    
    Paramètres
    ----------
    f : fonction R^n -> R
    V : numpy array( num )
        num: type numérique, longueur n.
 
    Retour
    ------
    numpy array( num )
    
    """
    n = len(P)
    grad = []
    for j in range(n):
        P_now = prepare_for_gradient( P , j )
        deriv_now = f(*P_now).d
        grad.append( deriv_now )
    return np.array(grad)
