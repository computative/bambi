import cython
import numpy as np

# benchmarked
class Charge:

    def __init__(self,q,r):
        self.q = q
        self.r = r

    # Check how many dimensions the carge is in cell
    def In(self, C):
        s = 0
        for i in range(3):
            s += int( C[0,i] <= self.r[i] <= C[1,i] )
        return True if s == 3 else False
    
# benchmarked
class Cell:
    
    def __init__(self,LBound,UBound):
        self.bounds = np.array([LBound,UBound]) 
        self.dim = np.array([
            UBound[0] - LBound[0],
            UBound[1] - LBound[1],
            UBound[2] - LBound[2]
        ])
        U, L = UBound, LBound
        self.vertices = np.array([
            [ U[0],L[1],L[2] ],
            [ L[0],U[1],L[2] ],
            [ L[0],L[1],U[2] ],
            [ U[0],U[1],L[2] ],
            [ U[0],L[1],U[2] ],
            [ L[0],U[1],U[2] ],
            [ U[0],U[1],U[2] ],
            [ L[0],L[1],L[2] ]])
        
        self.signature = np.array([
            [1,0,0],[0,1,0],[0,0,1],[1,1,0],
            [1,0,1],[0,1,1],[1,1,1],[0,0,0]])
        
    def __getitem__(self, tup):
        if type(tup) == tuple:
            x, y = tup
            return self.bounds[x][y]
        return self.bounds[tup]


def prune(Vertices, A):
    R = np.array([ An.r.tolist() for An in A ])
    q = np.array([ An.q for An in A ])
    # Finn den minimale distansen mellom partikler.
    D = np.array( [np.linalg.norm(ri-rj) for ri in R for rj in R] )
    DMin = np.min(D[D > 0])
    Cutoff = DMin/3
    Components = []
    table = np.zeros((len(Vertices),2))
    table[:,0] = np.arange(len(Vertices))
    label = 1
    particleLabel = []
    while 0 in table[:,1]:
        idx = np.argwhere( table[:,1] == 0 )[0,0]
        V = Vertices[idx]
        for _V, i  in zip(Vertices, range(len(Vertices)) ):
            if np.linalg.norm(_V - V) < Cutoff:
                table[i,1] = label
        for r in R:
            if np.linalg.norm(r - V) < Cutoff:
                particleLabel.append(label)
        label += 1
    roots = []; vals = []
    for Component in range(1,label):
        if Component not in particleLabel:
            ids = table[:,0][table[:,1] == Component].astype(int)
            V = Vertices[ids]
            _F = np.array([np.linalg.norm(F(_V,A)) for _V in V])
            argmin = np.argmin(_F)
            vals.append( np.linalg.norm(_F[argmin]) )
            roots.append( V[argmin] )
    return np.array(roots), np.array(vals)

    
# returnerer gradienten til potensialet
# benchmarked
def F(r,A):
    if r.shape == (3,1):
        r = r.T[0]
    arr = np.array([Ai.q*(r - Ai.r)*np.linalg.norm(r - Ai.r)**-3 for Ai in A ])
    return np.sum(arr,axis=0)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double BinMin(double x, double y):
    if x>y:
        return y
    else:
        return x


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double BinMax(double x, double y):
    if x<y:
        return y
    else:
        return x

    
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double max(double[:] x, int n):
    cdef int i = 1
    cdef double maximum = x[0]
    while i < n:
        if x[i] > maximum:
            maximum = x[i]
        i += 1
    return maximum


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double min(double[:] x, int n):
    cdef int i = 1
    cdef double minimum = x[0]
    while i < n:
        if x[i] > minimum:
            minimum = x[i]
        i += 1
    return minimum

# benchmarked
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double SupNorm(double[:] r, double[:,:] C):
    cdef double s = 0
    cdef int k = 0
    while k < 3:
        s += BinMax( (C[0,k]-r[k])**2,(C[1,k]-r[k])**2 )
        k += 1
    return s**.5


# benchmarked
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double InfNorm(double[:] r, double[:,:] C):
    cdef double s = 0
    cdef int k = 0
    while k < 3:
        if not( C[0,k] < r[k] < C[1,k] ):
            s += BinMin( (C[0,k]-r[k])**2, (C[1,k]-r[k])**2 )
        k += 1
    return s**.5


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double abs(double x):
    if x > -x:
        return x
    else:
        return -x

def FBounds(C, A):
    cdef double[:,:] _C = C[:]
    cdef double[:,:] _r = np.array([An.r for An in A])
    cdef double[:] _q = np.array([<double> An.q for An in A])
    cdef int _n = len(A)
    if not  _n >= 2:
        raise ValueError ("FBounds requires lists of at least two particles")
    if not( _r.shape[1] == 3 and _r.shape[0] > 1):
        raise ValueError ("FBounds requires at least two charges and in 3d space")
    if not (_C.shape[1] == 3 and _C.shape[0] == 2):
        raise ValueError ("First argument of FBounds() is not a valid Cell object")
    if not ( _q.shape[0] > 1 and _q.shape[0] == _r.shape[0]):
        raise ValueError ("Second argument of FBounds() is not a valid list of Charge objects")
    return _FBounds(_C,_r,_q,_n)


# Returnerer bounds pa Hessian
# testet at Hmin < Hmax
# here C refers to C.bounds
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:,:,:] _FBounds(double[:,:] C, double[:,:] r, double[:] q, int n):
    # dersom denne funksjonen kalles, vet vi at ingen
    # ladning er i cellcen C.
    cdef double H[2][3][3]
    cdef int tris[2]
    cdef int i,j,k
    cdef int bit, _bit, flag
    cdef double num
    # H[0] er HMin H[1] er HMax
    H[0][0][:] = [0, 0, 0]; H[0][1][:] = [0, 0, 0]; H[0][2][:] = [0, 0, 0]
    H[1][0][:] = [0, 0, 0]; H[1][1][:] = [0, 0, 0]; H[1][2][:] = [0, 0, 0]
    # loop over i>j for matrisen Hmax
    for i in range(3):
        for j in range(i,3):
            # dersom i=j, altsaa diagonal element
            if i == j:
                k = 0
                while k < n:
                    # dersom ladningen er positiv
                    if q[k] > 0:
                        # for Hmax
                        # dersom An sin i-te koord er i boksen C
                        #print(i,j,k)
                        if C[0,i] < r[k,i] < C[1,i]:
                            num = 0
                        else:
                            num = (C[1,i]-r[k,i])**2 if r[k,i] > C[1,i] else (C[0,i]-r[k,i])**2
                        H[<int>1][i][j] += q[k]/InfNorm(r[k,:],C)**3 - 3*q[k]*num/SupNorm(r[k,:],C)**5
                        # for Hmin
                        # dersom det i-te koordinate til An er
                        # over halvveis langs boksens i-te koord
                        if r[k,i] > (C[1,i] + C[0,i])/2:
                            num = (C[0,i] - r[k,i])**2
                        else:
                            num = (C[1,i] - r[k,i])**2
                        H[<int>0][i][j] += q[k]/SupNorm(r[k,:],C)**3 - 3*q[k]*num/InfNorm(r[k,:],C)**5
                    else: #dersom ladningen er negativ
                        # for Hmax
                        # dersom An sin i-te koord er i boksen C
                        num = (C[0,i]-r[k,i])**2 if r[k,i] > (C[0,i] + C[1,i])/2 else (C[1,i]-r[k,i])**2
                        H[<int>1][i][j] += q[k]/SupNorm(r[k,:],C)**3 - 3*q[k]*num/InfNorm(r[k,:],C)**5
                        # for Hmin
                        # dersom det i-te koordinate til An er
                        # over halvveis langs boksens i-te koord
                        if C[0,i] < r[k,i] < C[1,i]:
                            num = 0
                        else:
                            num = (C[1,i] - r[k,i])**2 if r[k,i] > C[1,i] else (C[0,i] - r[k,i])**2
                        H[<int>0][i][j] += q[k]/InfNorm(r[k,:],C)**3 - 3*q[k]*num/SupNorm(r[k,:],C)**5
                    k += 1
            else: # dersom i,j ikke indikerer diagonalelement
                k = 0
                while k < n:
                    #print("k2 =" , k)
                    tris[:] = [<int>(r[k,i] > C[1,i]) + <int>(r[k,i] > C[0,i]),
                               <int>(r[k,j] > C[1,j]) + <int>(r[k,j] > C[0,j])]
                    # dersom ladningen er positiv
                    if q[k] > 0:
                        # er folgende kode skrevet slik at C[1,:] er ovre skranke for C? 
                        if tris[0] != tris[1]:
                            num = (r[k,i] - C[<int>(tris[0]<=tris[1]),i] )*(r[k,j] - C[<int>(tris[0]>tris[1]),j])
                            H[<int>1][i][j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                            # dersom det er liten stor
                            if tris[0] + tris[1] == 2 and abs(tris[0] - tris[1]) == 2:
                                num = (r[k,i] - C[<int>(tris[0]//2),i])*(r[k,j] - C[<int>(tris[1]//2),j])
                                H[<int>0][i][j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                            else: # dersom det er (liten,medium) eller (medium,stor)
                                flag = 0 if (tris[0] == 2 or tris[1] == 2) else 1
                                num = (r[k,i] - C[flag,i])*(r[k,j] - C[flag,j])
                                H[<int>0][i][j] += -3*q[k]*num/SupNorm(r[k,:],C)**5
                        # begge er enten store eller smaa
                        elif tris[0] + tris[1] != 1: #ok
                            num = (C[tris[0]//2,i] - r[k,i])*(C[tris[1]//2,j] - r[k,j])
                            H[<int>1][i][j] += -3*q[k]*num/SupNorm(r[k,:],C)**5
                            num = (C[(1-tris[0]//2),i] - r[k,i])*(C[(1-tris[1]//2),j] - r[k,j])
                            H[<int>0][i][j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                        else: # dersom begge er medium #ok
                            num = BinMax( (C[0,i]-r[k,i])*(C[1,j] - r[k,j]) , (C[1,i] - r[k,i])*(C[0,j] - r[k,j]) )
                            H[<int>1][i][j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                            num = BinMin( (C[0,i] - r[k,i])*(C[0,j] - r[k,j]), (C[1,i] - r[k,i])*(C[1,j] - r[k,j]) )
                            H[<int>0][i][j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                    else: #dersom ladningen er negativ
                        if tris[0] == 1 and tris[1] == 1: #dersom An.r er i C langs baade i og j-aksen
                            H[<int>1][i][j] += -3*q[k]*BinMax((C[0,i] - r[k,i])*(C[0,j] - r[k,j]),
                                                     (C[1,i] - r[k,i])*(C[1,j] - r[k,j]))/InfNorm(r[k,:],C)**5
                            H[<int>0][i][j] += -3*q[k]*BinMax((C[0,i] - r[k,i])*(C[1,j] - r[k,j]),
                                                     (C[1,i] - r[k,i])*(C[0,j] - r[k,j]))/SupNorm(r[k,:],C)**5
                        elif tris[0] == tris[1]: # dersom Ak.r enten er stor eller liten
                            bit = tris[0]//2; _bit = <int>abs(<double>bit-1)
                            H[<int>1][i][j] += -3*q[k]*(C[_bit,i] - r[k,i])*(C[_bit,j] - r[k,j])/InfNorm(r[k,:],C)**5
                            H[<int>0][i][j] += -3*q[k]*(C[bit,i] - r[k,i])*(C[bit,j] - r[k,j])/SupNorm(r[k,:],C)**5
                        elif tris[0] + tris[1] == 2: # dersom den ene er stor, den andre liten
                            bit = tris[0]//2; _bit = <int>abs(<double>bit-1)
                            H[<int>1][i][j] += -3*q[k]*(C[bit,i] - r[k,i])*(C[_bit,j] - r[k,j])/SupNorm(r[k,:],C)**5
                            H[<int>0][i][j] += -3*q[k]*(C[_bit,i] - r[k,i])*(C[bit,j] - r[k,j])/InfNorm(r[k,:],C)**5
                        else: # dersom den ene stor, den andre medium eller den ene medium og den andre liten
                            if tris[0]*tris[1]//2: # dersom den ene stor og den andre medium
                                H[<int>1][i][j] += -3*q[k]*(C[0,i] - r[k,i])*(C[1,j] - r[k,j])/InfNorm(r[k],C)**5
                                H[<int>0][i][j] += -3*q[k]*(C[0,i] - r[k,i])*(C[0,j] - r[k,j])/InfNorm(r[k],C)**5
                            else: # dersom den ene medium og den andre liten
                                H[<int>1][i][j] += -3*q[k]*(C[1,i] - r[k,i])*(C[1,j] - r[k,j])/InfNorm(r[k,:],C)**5
                                H[<int>0][i][j] += -3*q[k]*(C[0,i] - r[k,i])*(C[1,j] - r[k,j])/InfNorm(r[k,:],C)**5
                    k += 1
    for i in range(1,3):
        for j in range(i):
            H[<int>1][i][j] = H[<int>1][j][i]
            H[<int>0][i][j] = H[<int>0][j][i]
    return H
