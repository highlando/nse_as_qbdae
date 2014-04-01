import dolfin
import scipy.sparse as sps
import numpy as np
from dolfin import dx

dolfin.parameters.linear_algebra_backend = "uBLAS"

N = 4
mesh = dolfin.UnitSquareMesh(N, N)

V = dolfin.FunctionSpace(mesh, 'CG', 2)

v = dolfin.TrialFunction(V)
u = dolfin.TestFunction(V)

dxnl, dynl = [], []
for i in range(V.dim()):
    bi = dolfin.Function(V)
    bvec = np.zeros((V.dim(), ))
    bvec[i] = 1
    bi.vector()[:] = bvec
    nxi = dolfin.assemble(u.dx(0)*bi*v*dx)
    nyi = dolfin.assemble(u.dx(1)*bi*v*dx)

    rows, cols, values = nxi.data()
    nxim = sps.csr_matrix((values, cols, rows))
    nxim.eliminate_zeros()
    dxnl.append(nxim)

    rows, cols, values = nyi.data()
    nyim = sps.csr_matrix((values, cols, rows))
    nyim.eliminate_zeros()
    dynl.append(nyim)

dxn = sps.hstack(dxnl)
dyn = sps.hstack(dynl)

hpart = sps.hstack([dxn, dyn])

NV = V.dim()
hmat = sps.vstack([sps.hstack([hpart, sps.csc_matrix((NV, NV**2))]),
                   sps.hstack([sps.csc_matrix((NV, NV**2)), hpart])])

# f = dolfin.Expression('1')
f = dolfin.Expression('x[0]*x[1]')
# f = dolfin.Expression('x[1]')
v1 = dolfin.interpolate(f,  V)

nform = dolfin.assemble(u.dx(0)*v1*v*dx)
rows, cols, values = nform.data()
nmat = sps.csr_matrix((values, cols, rows))

vvec = np.atleast_2d(v1.vector().array()).T

print np.linalg.norm(nmat*vvec)
print np.linalg.norm(dxn*np.kron(vvec, vvec))
print np.linalg.norm(nmat*vvec - dxn*np.kron(vvec, vvec))
