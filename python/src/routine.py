import numpy as np
from numpy.linalg import norm
import lib as py


# q is an array containing charges and pos
# contains the position vectors of each charge.
q = np.array([1,-1,-1,1,-1,1])
pos = np.array([[0.25,0,0],[-0.25,0,0],[0,-0.25,0],[0,0.25,0],[-0.1,0,0],[0.1,0,0]])
# maximal number of subdivisions
Itrs = 14
BoundingBox = np.array([[-0.5,-.5,-.5],[0.5,.5,.5]])
# list of charge objectxs
A = [py.Charge(qk,rk) for qk,rk in zip(q,pos)]

# sett ovre og nedre bounding box coord.
# Shifting the box by a small number
# ensures no charge counted twice
CListNext = [ py.Cell(BoundingBox[0] + [0.00001,-0.0002, 0.003],
                      BoundingBox[1] + [0.00000, 0.0010,-0.070]) ]
for i in range(Itrs):
    print(i+1)
    CList = CListNext; CListNext = []
    for C in CList: # for each cube
        # faa dimensjon til cellen C langs hver akse
        s = C.dim
        # sett foelgende initialverdiene for FMax og FMin
        # fordi dersom en ladning er i C, blir den
        # automatisk subdivided i andre if-test
        FMax = 1e7*np.ones(3); FMin = -1e7*np.ones(3)
        # regn andre skranker dersom ingen ladning er i C
        #print(array([Ak.In(C) for Ak in A])) 
        if not any(np.array([Ak.In(C) for Ak in A])):
            # Hmin er nullte index, Hmax er foerste index
            H = py.FBounds(C,A)
            #print(H[0][0][0],H[0][0][1],H[0][0][2])
            k = 0
            for V, flag in zip(C.vertices,C.signature):
                sgn = 1-2*flag
                FMaxTemp = py.F(V,A) + np.maximum(0, sgn[0]*s[0]*H[1-flag[0]][:,0] ) + \
                                       np.maximum(0, sgn[1]*s[1]*H[1-flag[1]][:,1] ) + \
                                       np.maximum(0, sgn[2]*s[2]*H[1-flag[2]][:,2] )
                FMinTemp = py.F(V,A) + np.minimum(0, sgn[0]*s[0]*H[ flag[0] ][:,0] ) + \
                                       np.minimum(0, sgn[1]*s[1]*H[ flag[1] ][:,1] ) + \
                                       np.minimum(0, sgn[2]*s[2]*H[ flag[2] ][:,2] )
                FMax = np.minimum(FMaxTemp,FMax)
                FMin = np.maximum(FMinTemp,FMin)
                k += 1
        # subdivide dersom det muligens er en null
        if (not any(FMin*FMax>0)):
            if Itrs - i > 1:
                [a1,b1,c1], [a2,b2,c2] = C[:]
                _5a, _5b, _5c = 0.5*(a1+a2), 0.5*(b1+b2), 0.5*(c1+c2)
                # subdivide
                CListNext.extend(
                    [py.Cell(np.array([_5a,_5b,c1]),np.array([a2,b2,_5c])),
                     py.Cell(np.array([a1,_5b,c1]),np.array([_5a,b2,_5c])),
                     py.Cell(np.array([a1,_5b,_5c]),np.array([_5a,b2,c2])),
                     py.Cell(np.array([a1,b1,_5c]),np.array([_5a,_5b,c2])),
                     py.Cell(np.array([_5a,b1,_5c]),np.array([a2,_5b,c2])),
                     py.Cell(np.array([_5a,b1,c1]),np.array([a2,_5b,_5c])),
                     py.Cell(np.array([a1,b1,c1]),np.array([_5a,_5b,_5c])),
                     py.Cell(np.array([_5a,_5b,_5c]),np.array([a2,b2,c2]))]
                )
            else:
                CListNext.append(C)
            

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(10, 10))
axs = fig.add_subplot(111, projection='3d')

axs.set_xlim(-.5, .5)
axs.set_ylim(-.5, .5)
axs.set_zlim(-.5, .5)

PooledList = []
print("Pruning list...")
for i in range(len(CListNext)):
    for C in CListNext:
        PooledList.extend([V.tolist() for V in C.vertices])
vertices = np.unique(np.array(PooledList), axis = 0)
roots, vals = py.prune(vertices,A)
print("Zeros found:")
print(roots)

axs.scatter(roots[:,0], roots[:,1], roots[:,2])
axs.scatter(pos[:,0], pos[:,1], pos[:,2], ('o', -30, -5))

for root,val in zip(roots,vals):
    axs.text(root[0], root[1], root[2], "%.3f" % val, "x",fontsize=8)

plt.tight_layout()
plt.show()
