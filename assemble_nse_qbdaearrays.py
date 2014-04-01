import dolfin
import scipy.sparse as sps
import numpy as np
from dolfin import dx, grad, inner

dolfin.parameters.linear_algebra_backend = "uBLAS"

N = 6
mesh = dolfin.UnitSquareMesh(N, N)

V = dolfin.FunctionSpace(mesh, 'CG', 2)
W = dolfin.VectorFunctionSpace(mesh, 'CG', 2)

v = dolfin.TrialFunction(V)
vt = dolfin.TestFunction(V)

w = dolfin.TrialFunction(W)
wt = dolfin.TestFunction(W)

dxnl, dynl = [], []
for i in range(V.dim()):
    bi = dolfin.Function(V)
    bvec = np.zeros((V.dim(), ))
    bvec[i] = 1
    bi.vector()[:] = bvec
    nxi = dolfin.assemble(v.dx(0)*bi*vt*dx)
    nyi = dolfin.assemble(v.dx(1)*bi*vt*dx)

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
hmat = sps.vstack([sps.hstack([hpart, sps.csc_matrix((NV, 2*NV**2))]),
                   sps.hstack([sps.csc_matrix((NV, 2*NV**2)), hpart])])

# f = dolfin.Expression('1')
# xexp = 'x[0]*x[1]'
# yexp = 'x[0]*x[1]'

xexp = 'x[1]'
yexp = '0'

f = dolfin.Expression((xexp, yexp))
fx = dolfin.Expression(xexp)
fy = dolfin.Expression(yexp)

u = dolfin.interpolate(f,  W)
ux = dolfin.interpolate(fx,  V)
uy = dolfin.interpolate(fy,  V)

nform = dolfin.assemble(inner(grad(w)*u, wt) * dx)
rows, cols, values = nform.data()
nmat = sps.csr_matrix((values, cols, rows))

uvec = np.atleast_2d(u.vector().array()).T
uvecx = np.atleast_2d(ux.vector().array()).T
uvecy = np.atleast_2d(uy.vector().array()).T
uvecxy = np.vstack([uvecx, uvecy])

print np.linalg.norm(nmat*uvec)
print np.linalg.norm(hmat*np.kron(uvecxy, uvecxy))
# print np.linalg.norm(nmat*uvec - hmat*np.kron(uvecxy, uvecxy))
