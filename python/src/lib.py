from numpy import *
from numpy.linalg import norm

# benchmarked
class Charge:
    def __init__(self,q,r):
        self.q = q
        self.r = r

    # Check if the charge is in the cell C
    def In(self,C):
        return all(logical_and((C[0,:] <= self.r),(self.r <= C[1,:])))

# benchmarked
# benchmarked
class Cell:
    
    def __init__(self,LBound,UBound):
        self.bounds = array([LBound,UBound]) 
        self.dim = array([
            UBound[0] - LBound[0],
            UBound[1] - LBound[1],
            UBound[2] - LBound[2]
        ])
        U, L = UBound, LBound
        self.vertices = array([
            [ U[0],L[1],L[2] ],
            [ L[0],U[1],L[2] ],
            [ L[0],L[1],U[2] ],
            [ U[0],U[1],L[2] ],
            [ U[0],L[1],U[2] ],
            [ L[0],U[1],U[2] ],
            [ U[0],U[1],U[2] ],
            [ L[0],L[1],L[2] ]])
        
        self.signature = array([
            [1,0,0],[0,1,0],[0,0,1],[1,1,0],
            [1,0,1],[0,1,1],[1,1,1],[0,0,0]])
        
    def __getitem__(self, tup):
        if type(tup) == tuple:
            x, y = tup
            return self.bounds[x][y]
        return self.bounds[tup]


def prune(Vertices, A):
    R = array([ An.r.tolist() for An in A ])
    q = array([ An.q for An in A ])
    # Finn den minimale distansen mellom partikler.
    D = array( [linalg.norm(ri-rj) for ri in R for rj in R] )
    DMin = min(D[D > 0])
    Cutoff = DMin/3
    Components = []
    table = zeros((len(Vertices),2))
    table[:,0] = arange(len(Vertices))
    label = 1
    particleLabel = []
    while 0 in table[:,1]:
        idx = argwhere( table[:,1] == 0 )[0,0]
        V = Vertices[idx]
        for _V, i  in zip(Vertices, range(len(Vertices)) ):
            if linalg.norm(_V - V) < Cutoff:
                table[i,1] = label
        for r in R:
            if linalg.norm(r - V) < Cutoff:
                particleLabel.append(label)
        label += 1
    roots = []; vals = []
    for Component in range(1,label):
        if Component not in particleLabel:
            ids = table[:,0][table[:,1] == Component].astype(int)
            V = Vertices[ids]
            _F = array([linalg.norm(F(_V,A)) for _V in V])
            _arg = argmin(_F)
            vals.append( linalg.norm(_F[_arg]) )
            roots.append( V[_arg] )
    return array(roots), array(vals)


# benchmarked
def InfNorm(r,C):
    s = 0
    for k in setdiff1d(arange(3), argwhere(logical_and((C[0,:] < r),(r < C[1,:]))).T[0] ):
        s += min( (C[0,k]-r[k])**2,(C[1,k]-r[k])**2 ) 
    return s**.5

# benchmarked
def SupNorm(r,C):
    s = 0
    for k in range(3):
        s += max( (C[0,k]-r[k])**2,(C[1,k]-r[k])**2 ) 
    return s**.5

# returnerer gradienten til potensialet
# benchmarked
def F(r,A):
    if r.shape == (3,1):
        r = r.T[0]
    arr = array([Ai.q*(r - Ai.r)*linalg.norm(r - Ai.r)**-3 for Ai in A ])
    return sum(arr,axis=0)


# Returnerer bounds pa Hessian
# testet at Hmin < Hmax
def FBounds(C, A):
    # dersom denne funksjonen kalles, vet vi at ingen
    # ladning er i cellcen C.
    HMax = zeros((3,3))
    HMin = zeros((3,3))
    # loop over i>j for matrisen Hmax
    for i in range(3):
        for j in range(i,3):
            # dersom i=j, altsaa diagonal element
            if i == j:
                for An in A:
                    # dersom ladningen er positiv
                    if An.q > 0:
                        # for Hmax
                        # dersom An sin i-te koord er i boksen C
                        if C[0,i] < An.r[i] < C[1,i]:
                            num = 0
                        else:
                            num = (C[1,i]-An.r[i])**2 if An.r[i] > C[1,i] else (C[0,i]-An.r[i])**2
                        HMax[i,j] += An.q/InfNorm(An.r,C)**3 - 3*An.q*num/SupNorm(An.r,C)**5
                        # for Hmin
                        # dersom det i-te koordinate til An er
                        # over halvveis langs boksens i-te koord
                        if An.r[i] > (C[1,i] + C[0,i])/2:
                            num = (C[0,i] - An.r[i])**2
                        else:
                            num = (C[1,i] - An.r[i])**2
                        HMin[i,j] += An.q/SupNorm(An.r,C)**3 - 3*An.q*num/InfNorm(An.r,C)**5
                    else: #dersom ladningen er negativ
                        # for Hmax
                        # dersom An sin i-te koord er i boksen C
                        num = (C[0,i]-An.r[i])**2 if An.r[i] > (C[0,i] + C[1,i])/2 else (C[1,i]-An.r[i])**2
                        HMax[i,j] += An.q/SupNorm(An.r,C)**3 - 3*An.q*num/InfNorm(An.r,C)**5
                        # for Hmin
                        # dersom det i-te koordinate til An er
                        # over halvveis langs boksens i-te koord
                        if C[0,i] < An.r[i] < C[1,i]:
                            num = 0
                        else:
                            num = (C[1,i] - An.r[i])**2 if An.r[i] > C[1,i] else (C[0,i] - An.r[i])**2
                        HMin[i,j] += An.q/InfNorm(An.r,C)**3 - 3*An.q*num/SupNorm(An.r,C)**5
            else: # dersom i,j ikke indikerer diagonalelement
                for An in A:
                    tris = [int(An.r[i] > C[1,i]) + int(An.r[i] > C[0,i]),
                            int(An.r[j] > C[1,j]) + int(An.r[j] > C[0,j])]
                    # dersom ladningen er positiv
                    if An.q > 0:
                        # er folgende kode skrevet slik at C[1,:] er ovre skranke for C? 
                        if tris[0] != tris[1]:
                            num = (An.r[i] - C[int(tris[0]<=tris[1]),i] )*(An.r[j] - C[int(tris[0]>tris[1]),j])
                            HMax[i,j] += -3*An.q*num/InfNorm(An.r,C)**5
                            # dersom det er liten stor
                            if tris[0] + tris[1] == 2 and abs(tris[0] - tris[1]) == 2:
                                num = (An.r[i] - C[int(tris[0]//2),i])*(An.r[j] - C[int(tris[1]//2),j])
                                HMin[i,j] += -3*An.q*num/InfNorm(An.r,C)**5
                            else: # dersom det er (liten,medium) eller (medium,stor)
                                flag = 0 if 2 in tris else 1
                                num = (An.r[i] - C[flag,i])*(An.r[j] - C[flag,j])
                                HMin[i,j] += -3*An.q*num/SupNorm(An.r,C)**5
                        # begge er enten store eller smaa
                        elif tris[0] + tris[1] != 1: #ok
                            num = (C[tris[0]//2,i] - An.r[i])*(C[tris[1]//2,j] - An.r[j])
                            HMax[i,j] += -3*An.q*num/SupNorm(An.r,C)**5
                            num = (C[(1-tris[0]//2),i] - An.r[i])*(C[(1-tris[1]//2),j] - An.r[j])
                            HMin[i,j] += -3*An.q*num/InfNorm(An.r,C)**5
                        else: # dersom begge er medium #ok
                            num = max( (C[0,i]-An.r[i])*(C[1,j] - An.r[j]) , (C[1,i] - An.r[i])*(C[0,j] - An.r[j]) )
                            HMax[i,j] += -3*An.q*num/InfNorm(An.r,C)**5
                            num = min( (C[0,i] - An.r[i])*(C[0,j] - An.r[j]), (C[1,i] - An.r[i])*(C[1,j] - An.r[j]) )
                            HMin[i,j] += -3*An.q*num/InfNorm(An.r,C)**5
                    else: #dersom ladningen er negativ
                        if tris[0] == 1 and tris[1] == 1: #dersom An.r er i C langs baade i og j-aksen
                            HMax[i,j] += -3*An.q*max((C[0,i] - An.r[i])*(C[0,j] - An.r[j]),
                                                     (C[1,i] - An.r[i])*(C[1,j] - An.r[j]))/InfNorm(An.r,C)**5
                            HMin[i,j] += -3*An.q*max((C[0,i] - An.r[i])*(C[1,j] - An.r[j]),
                                                     (C[1,i] - An.r[i])*(C[0,j] - An.r[j]))/SupNorm(An.r,C)**5
                        elif tris[0] == tris[1]: # dersom Ak.r enten er stor eller liten
                            bit = tris[0]//2; _bit = abs(bit-1)
                            HMax[i,j] += -3*An.q*(C[_bit,i] - An.r[i])*(C[_bit,j] - An.r[j])/InfNorm(An.r,C)**5
                            HMin[i,j] += -3*An.q*(C[bit,i] - An.r[i])*(C[bit,j] - An.r[j])/SupNorm(An.r,C)**5
                        elif tris[0] + tris[1] == 2: # dersom den ene er stor, den andre liten
                            bit = tris[0]//2; _bit = abs(bit-1)
                            HMax[i,j] += -3*An.q*(C[bit,i] - An.r[i])*(C[_bit,j] - An.r[j])/SupNorm(An.r,C)**5
                            HMin[i,j] += -3*An.q*(C[_bit,i] - An.r[i])*(C[bit,j] - An.r[j])/InfNorm(An.r,C)**5
                        else: # dersom den ene stor, den andre medium eller den ene medium og den andre liten
                            if tris[0]*tris[1]//2: # dersom den ene stor og den andre medium
                                HMax[i,j] += -3*An.q*(C[0,i] - An.r[i])*(C[1,j] - An.r[j])/InfNorm(An.r,C)**5
                                HMin[i,j] += -3*An.q*(C[0,i] - An.r[i])*(C[0,j] - An.r[j])/InfNorm(An.r,C)**5
                            else: # dersom den ene medium og den andre liten
                                HMax[i,j] += -3*An.q*(C[1,i] - An.r[i])*(C[1,j] - An.r[j])/InfNorm(An.r,C)**5
                                HMin[i,j] += -3*An.q*(C[0,i] - An.r[i])*(C[1,j] - An.r[j])/InfNorm(An.r,C)**5
    for i in range(1,3):
        for j in range(i):
            HMax[i,j] = HMax[j,i]
            HMin[i,j] = HMin[j,i]
    return array([HMin, HMax])

