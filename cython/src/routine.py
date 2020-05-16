import numpy as np
import lib as cy
import multiprocessing as mp


############################################################################################
#                                   INSTRUCTIONS START                                     #
############################################################################################

# This script computes the zeros of the electric field of a collection of charges. The
# method follows:
#
#    Max, N. and Weinkauf, T. Critical Points of the Electric Field from a Collection of
#  Point Charges. in Topology-Based Methods in Visualization II 101–114 (Springer, 2009).
#
# The script requires positions and charges of of two or more particles to be provided
# by the end user. The coordinates are specified by entering them [x,y,z] one row for
# each particle in the matrix r and the array q holds the charges for particles 1, 2, ...
# etc. The units are hartree atomic units

r = np.array([
    [ 0.25, 0.00, 0.00],
    [-0.25, 0.00, 0.00],
    [ 0.00,-0.25, 0.00],
    [ 0.00, 0.25, 0.00],
    [-0.10, 0.00, 0.00],
    [ 0.10, 0.00, 0.00]
])

q = np.array([-1.,-1.,-1.,5.,-1.,-1.])

# the script runs a specified number of iterations. If the number of iterations are too
# low, the script will result in odd errors or meaningless results. A good starting point
# may be 10 iterations. Increase iterations for increased accuracy.

Itrs = 20

# The script searches inside a bounding box that you specify. To specify the box, you
# provide exactly two sets of three coordinates. The first 3 coordinates (x1,y1,z1) specify
# the lower bound of the box. The last 3 coordinates (x2,y2,z2) are the upper bounds on the 
# box. In mathematics, this correspond to the cartesian product [x1,x2]x[y1,y2]x[z1,z2].

BoundingBox = np.array([[-0.5,-.5,-.5],[0.5,.5,.5]])

############################################################################################
#                                   INSTRUCTIONS END                                       #
############################################################################################

# the following lines are the implementation of the method. If they are edited, it likely
#breaks.

# this is a list of Charge objects.
A = [cy.Charge(qk,rk) for qk,rk in zip(q,r)]
n = len(A)

# The bounding box should be shifted by a small amount to essentially eliminate the possibility
# that a charge is located on the boundary of two cells in a future subdivision. This improves
# stability of the program for some reason, and has no consequences for accuracy.

CListNext = [ cy.Cell(BoundingBox[0] + [ 0.00001,-0.00002, 0.0003],
                      BoundingBox[1] + [-0.00004, 0.00010,-0.0070]) ]

# the code is parallelisable, and the following is the main routine for each processor that
# estimates if there is a zero in some given cell. If one cannot be sure that a zero is inside
# the cell, the cell is subdivided.

def burden(args):
    C, I = args
    s = C.dim
    # set some large initial values of FMin and FMmax
    FMax = 1e31*np.ones(3); FMin = -1e31*np.ones(3)
    # if there is no charge in the cell C  then...
    if not np.any(np.array([Ak.In(C) for Ak in A])):
        # calculate Hessian HMin and HMax (contained in H[0], H[1] respectively
        H = cy.FBounds(C,r,q,n)
        # for each vertex in the cell C ...
        for V, flag in zip(C.vertices,C.signature):
            sgn = 1-2*flag
            # get an estimate for FMin and FMax
            F = cy.F(V,r,q,n)
            FMaxTemp = F + np.maximum( 0, sgn[0]*s[0]*H[1-flag[0]][0,:] ) + \
                           np.maximum( 0, sgn[1]*s[1]*H[1-flag[1]][1,:] ) + \
                           np.maximum( 0, sgn[2]*s[2]*H[1-flag[2]][2,:] )
            FMinTemp = F + np.minimum( 0, sgn[0]*s[0]*H[ flag[0] ][0,:] ) + \
                           np.minimum( 0, sgn[1]*s[1]*H[ flag[1] ][1,:] ) + \
                           np.minimum( 0, sgn[2]*s[2]*H[ flag[2] ][2,:] )
            # do componentwise min/max with respect to the previous iteration (if any)
            FMax = np.minimum( FMaxTemp, FMax )
            FMin = np.maximum( FMinTemp, FMin )
    # If we cannot be sure that there is no zero in the cell C ...
    if (not np.any(FMin*FMax>0)):
        # ... and it is not the last iteration
        if Itrs - I > 1:
            [a1,b1,c1], [a2,b2,c2] = C[:]
            _5a, _5b, _5c = 0.5*(a1+a2), 0.5*(b1+b2), 0.5*(c1+c2)
            # subdivide the cell C
            return [cy.Cell(np.array([_5a,_5b,c1]),np.array([a2,b2,_5c])),
                    cy.Cell(np.array([a1,_5b,c1]),np.array([_5a,b2,_5c])),
                    cy.Cell(np.array([a1,_5b,_5c]),np.array([_5a,b2,c2])),
                    cy.Cell(np.array([a1,b1,_5c]),np.array([_5a,_5b,c2])),
                    cy.Cell(np.array([_5a,b1,_5c]),np.array([a2,_5b,c2])),
                    cy.Cell(np.array([_5a,b1,c1]),np.array([a2,_5b,_5c])),
                    cy.Cell(np.array([a1,b1,c1]),np.array([_5a,_5b,_5c])),
                    cy.Cell(np.array([_5a,_5b,_5c]),np.array([a2,b2,c2]))]
        else:
            return [C]
        # and if not, bring the cell C to the next iteration
    return None

for i in range(Itrs):
    print(i)
    CList = CListNext; CListNext = []
    # for each Cell object in CList
    args = [ (C,i) for C in CList  ]
    for Cells in mp.Pool().map(burden, args):
        if Cells != None:
            CListNext.extend(Cells)

# most vertices are expected to appear exactly 8 times in the list of cells
# because each vertex has 8 neighbouring cells. We quotient out these 
# redundancies and ...

j = 0
PooledList = []
print("Pruning list...")
for C in CListNext:
    PooledList.extend([V.tolist() for V in C.vertices])
verts = np.unique(np.array(PooledList), axis = 0)


# ... and merge vertices that are sufficiently close together to a single point
# for which the value of F is calculated exactly. This will be our estimate for
# the zero of of F. Keep in mind that the program doesn't discern between a 
# cell containing a charge and a zero. So we delete vertices that are 'sufficiently'
# close to a charge.

d = np.max(np.array([ sum((C.dim)**2)**.5 for C in CListNext ]))
roots, vals = cy.prune(verts,A,d)

# the program prints out the zeros that were found:

print("Zeros found:")
print(roots)

# and attempts to plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(10, 10))
axs = fig.add_subplot(111, projection='3d')

axs.set_xlim(-.5, .5)
axs.set_ylim(-.5, .5)
axs.set_zlim(-.5, .5)

axs.scatter(roots[:,0], roots[:,1], roots[:,2])
axs.scatter(r[:,0], r[:,1], r[:,2], ('o', -30, -5))

for root,val in zip(roots,vals):
    axs.text(root[0], root[1], root[2], "%.3f" % val, "x",fontsize=8)

plt.tight_layout()
axs.set_xlabel(r"x [bohr]")
axs.set_ylabel(r"y [bohr]")
axs.set_zlabel(r"z [bohr]")
plt.tight_layout(1)
plt.savefig("output.png")
