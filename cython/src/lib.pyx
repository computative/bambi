import cython
import numpy as np

# this is a class representing a charge
class Charge:
    def __init__(self,q,r):
        self.q = q
        self.r = r

    # this function checks if the present charge is in the cell C
    def In(self, C):
        s = 0
        for i in range(3):
            s += int( C[0,i] <= self.r[i] <= C[1,i] )
        return True if s == 3 else False
    
# this class represents a box in 3 dimensional euclidean space
class Cell:
    def __init__(self,LBound,UBound):
        # self.bounds contains a lower and upper bound of the box.
        # if the lower bound is (x1,y1,z1) and the upper bound is
        # (x2,y2,z2), this sets the present cell to be the cartesian
        # product [x1,x2]x[y1,y2]x[z1,z2]
        self.bounds = np.array([LBound,UBound])
        # this variable contains the width, length and height of the cell 
        self.dim = np.array([
            UBound[0] - LBound[0],
            UBound[1] - LBound[1],
            UBound[2] - LBound[2]
        ])
        U, L = UBound, LBound
        # this array contains the coordinates of the vertices of the box.
        self.vertices = np.array([
            [ U[0],L[1],L[2] ],
            [ L[0],U[1],L[2] ],
            [ L[0],L[1],U[2] ],
            [ U[0],U[1],L[2] ],
            [ U[0],L[1],U[2] ],
            [ L[0],U[1],U[2] ],
            [ U[0],U[1],U[2] ],
            [ L[0],L[1],L[2] ]])
        # this array contains a boolean representation of whether it is True
        # or False that a specific coordinate of a vertex is an upper or lower
        # bound in the sense that we defined it above. 
        self.signature = np.array([
            [1,0,0],[0,1,0],[0,0,1],[1,1,0],
            [1,0,1],[0,1,1],[1,1,1],[0,0,0]])
    
    # self explainatory
    def __getitem__(self, tup):
        if type(tup) == tuple:
            x, y = tup
            return self.bounds[x][y]
        return self.bounds[tup]

# this function determines the smallest possible sphere that contains a component
# of the separation (in the topological sense) [of vertices] centered at r. It then
# returns the indices of the vertices inside the sphere at the output

def scoping(r,vertices,diag):
    n = len(vertices)
    m = 1
    mp = 0
    k = 0
    while m > mp:
        k += 1
        component = []
        for i in range(n):
            if np.linalg.norm(r - vertices[i]) < (4/3+k)*diag:
                component.append(i)
        mp = m
        m = len(component)
    return component

# this function removes the components [of vertices] centered at each charge in A and
# then takes the components containing a zero, and returns the smallest values of
# the euclidean norm of F at that point. This is the estimate for the zero.

def prune(vertices,A,diag):
    r = np.array([ An.r.tolist() for An in A ])
    q = np.array([ An.q for An in A ])
    n = len(q)
    chrg_comp = []
    for _r in r:
        chrg_comp.extend(scoping(_r, vertices, diag))

    verts = []
    vals = []
    num_nonchrg_verts = 0
    for i, V in zip(range(len(vertices)), vertices):
        if i not in chrg_comp:
            num_nonchrg_verts += 1
            verts.append(V)
            vals.append(np.linalg.norm(F(V,r,q,n)))
    vals = np.array(vals); verts = np.array(verts)

    i = j = 0
    priority = np.argsort( vals )
    outzeros = []; outvalues = []; outidx = []
    
    while i < num_nonchrg_verts:
        if priority[j] in outidx:
            j += 1
            continue
        component = scoping( verts[priority][j], verts, diag )
        outzeros.append( verts[priority][j] )
        outvalues.append( vals[priority][j] )
        outidx.extend( component )
        j += 1
        i += len(component)

    return np.array(outzeros), np.array(outvalues)


# a cdef-function that returns the gradient of the potential
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double[::1] _F(double[::1] r, double[:,::1] Ar, double[::1] Aq, long n):
    cdef double[::1] s = np.zeros(3)
    cdef double norm
    for i in range(n):
        norm = 0
        for j in range(3):
            norm += (r[j] - Ar[i,j])**2
        for j in range(3):
            s[j] += Aq[i]*(r[j] - Ar[i,j])/(norm)**(1.5)
    return s


# a python function that returns the gradient of the potential
def F(r,Ar,Aq,n):
    return np.array(_F(r,Ar,Aq,n))

# this function finds the minimum of two numbers...
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double BinMin(double x, double y):
    if x>y:
        return y
    else:
        return x

# ...and this function finds the maximum
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double BinMax(double x, double y):
    if x<y:
        return y
    else:
        return x

# This function finds the supremum of the distance from r to C...
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double SupNorm(double[:] r, double[:,:] C):
    cdef double s = 0
    cdef int k = 0
    while k < 3:
        s += BinMax( (C[0,k]-r[k])**2,(C[1,k]-r[k])**2 )
        k += 1
    return s**.5


# ... and this function finds the infimum of the same quantity..
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double InfNorm(double[:] r, double[:,:] C):
    cdef double s = 0
    cdef int k = 0
    while k < 3:
        if not( C[0,k] < r[k] < C[1,k] ):
            s += BinMin( (C[0,k]-r[k])**2, (C[1,k]-r[k])**2 )
        k += 1
    return s**.5


# this function returns the absolute value of a number.
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double abs(double x):
    if x > -x:
        return x
    else:
        return -x

# this is a python function that interfaces with the cdef-function of a
# similar name. this function conditions the variables that the C-function
# uses and perform bounds checks and mitigate the risk of a zero division.
def FBounds(C, double[:,::1] Ar, double[::1] Aq, int n):
    cdef double[:,::1] _C = C[:]
    return np.array(_FBounds(_C,Ar,Aq,n))


# this is a cdef-function that computes the lower and upper bounds of the hessian H
# it is really technical, so you wouldn't get much out of reading it. What it does
# is fastest explained by Max, N. and Weinkauf (2009).
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double[:,:,::1] _FBounds(double[:,::1] C, double[:,::1] r, double[::1] q, int n):
    cdef double[:,:,::1] H = np.zeros((2,3,3))
    cdef int tris0,tris1
    cdef int i,j,k
    cdef int bit, _bit, flag
    cdef double num
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
                        if C[0,i] < r[k,i] < C[1,i]:
                            num = 0
                        else:
                            num = (C[1,i]-r[k,i])**2 if r[k,i] > C[1,i] else (C[0,i]-r[k,i])**2
                        H[1,i,j] += q[k]/InfNorm(r[k,:],C)**3 - 3*q[k]*num/SupNorm(r[k,:],C)**5
                        # for Hmin
                        # dersom det i-te koordinate til An er
                        # over halvveis langs boksens i-te koord
                        if r[k,i] > (C[1,i] + C[0,i])/2:
                            num = (C[0,i] - r[k,i])**2
                        else:
                            num = (C[1,i] - r[k,i])**2
                        H[0,i,j] += q[k]/SupNorm(r[k,:],C)**3 - 3*q[k]*num/InfNorm(r[k,:],C)**5
                    else: #dersom ladningen er negativ
                        # for Hmax
                        # dersom An sin i-te koord er i boksen C
                        num = (C[0,i]-r[k,i])**2 if r[k,i] > (C[0,i] + C[1,i])/2 else (C[1,i]-r[k,i])**2
                        H[1,i,j] += q[k]/SupNorm(r[k,:],C)**3 - 3*q[k]*num/InfNorm(r[k,:],C)**5
                        # for Hmin
                        # dersom det i-te koordinate til An er
                        # over halvveis langs boksens i-te koord
                        if C[0,i] < r[k,i] < C[1,i]:
                            num = 0
                        else:
                            num = (C[1,i] - r[k,i])**2 if r[k,i] > C[1,i] else (C[0,i] - r[k,i])**2
                        H[0,i,j] += q[k]/InfNorm(r[k,:],C)**3 - 3*q[k]*num/SupNorm(r[k,:],C)**5
                    k += 1
            else: # dersom i,j ikke indikerer diagonalelement
                k = 0
                while k < n:
                    #print("k2 =" , k)
                    tris0 = <int>(r[k,i] > C[1,i]) + <int>(r[k,i] > C[0,i])
                    tris1 = <int>(r[k,j] > C[1,j]) + <int>(r[k,j] > C[0,j])
                    # dersom ladningen er positiv
                    if q[k] > 0:
                        # er folgende kode skrevet slik at C[1,:] er ovre skranke for C? 
                        if tris0 != tris1:
                            num = (r[k,i] - C[<int>(tris0<=tris1),i] )*(r[k,j] - C[<int>(tris0>tris1),j])
                            H[1,i,j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                            # dersom det er liten stor
                            if tris0 + tris1 == 2 and abs(tris0 - tris1) == 2:
                                num = (r[k,i] - C[<int>tris0//2,i])*(r[k,j] - C[<int>tris1//2,j])
                                H[0,i,j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                            else: # dersom det er (liten,medium) eller (medium,stor)
                                flag = 0 if (tris0 == 2 or tris1 == 2) else 1
                                num = (r[k,i] - C[flag,i])*(r[k,j] - C[flag,j])
                                H[0,i,j] += -3*q[k]*num/SupNorm(r[k,:],C)**5
                        # begge er enten store eller smaa
                        elif tris0 + tris1 != 1: #ok
                            num = (C[tris0//2,i] - r[k,i])*(C[tris1//2,j] - r[k,j])
                            H[1,i,j] += -3*q[k]*num/SupNorm(r[k,:],C)**5
                            num = (C[(1-tris0//2),i] - r[k,i])*(C[(1-tris1//2),j] - r[k,j])
                            H[0,i,j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                        else: # dersom begge er medium #ok
                            num = BinMax( (C[0,i]-r[k,i])*(C[1,j] - r[k,j]) , (C[1,i] - r[k,i])*(C[0,j] - r[k,j]) )
                            H[1,i,j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                            num = BinMin( (C[0,i] - r[k,i])*(C[0,j] - r[k,j]), (C[1,i] - r[k,i])*(C[1,j] - r[k,j]) )
                            H[0,i,j] += -3*q[k]*num/InfNorm(r[k,:],C)**5
                    else: #dersom ladningen er negativ
                        if tris0 == 1 and tris1 == 1: #dersom An.r er i C langs baade i og j-aksen
                            H[1,i,j] += -3*q[k]*BinMax((C[0,i] - r[k,i])*(C[0,j] - r[k,j]),
                                                     (C[1,i] - r[k,i])*(C[1,j] - r[k,j]))/InfNorm(r[k,:],C)**5
                            H[0,i,j] += -3*q[k]*BinMax((C[0,i] - r[k,i])*(C[1,j] - r[k,j]),
                                                     (C[1,i] - r[k,i])*(C[0,j] - r[k,j]))/SupNorm(r[k,:],C)**5
                        elif tris0 == tris1: # dersom Ak.r enten er stor eller liten
                            bit = tris0//2; _bit = <int>abs(<double>bit-1)
                            H[1,i,j] += -3*q[k]*(C[_bit,i] - r[k,i])*(C[_bit,j] - r[k,j])/InfNorm(r[k,:],C)**5
                            H[0,i,j] += -3*q[k]*(C[bit,i] - r[k,i])*(C[bit,j] - r[k,j])/SupNorm(r[k,:],C)**5
                        elif tris0 + tris1 == 2: # dersom den ene er stor, den andre liten
                            bit = tris0//2; _bit = <int>abs(<double>bit-1)
                            H[1,i,j] += -3*q[k]*(C[bit,i] - r[k,i])*(C[_bit,j] - r[k,j])/SupNorm(r[k,:],C)**5
                            H[0,i,j] += -3*q[k]*(C[_bit,i] - r[k,i])*(C[bit,j] - r[k,j])/InfNorm(r[k,:],C)**5
                        else: # dersom den ene stor, den andre medium eller den ene medium og den andre liten
                            if tris0*tris1//2: # dersom den ene stor og den andre medium
                                H[1,i,j] += -3*q[k]*(C[0,i] - r[k,i])*(C[1,j] - r[k,j])/InfNorm(r[k],C)**5
                                H[0,i,j] += -3*q[k]*(C[0,i] - r[k,i])*(C[0,j] - r[k,j])/InfNorm(r[k],C)**5
                            else: # dersom den ene medium og den andre liten
                                H[1,i,j] += -3*q[k]*(C[1,i] - r[k,i])*(C[1,j] - r[k,j])/InfNorm(r[k,:],C)**5
                                H[0,i,j] += -3*q[k]*(C[0,i] - r[k,i])*(C[1,j] - r[k,j])/InfNorm(r[k,:],C)**5
                    k += 1
    for i in range(1,3):
        for j in range(i):
            H[1,i,j] = H[1,j,i]
            H[0,i,j] = H[0,j,i]
    return H
