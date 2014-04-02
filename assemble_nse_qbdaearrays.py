import dolfin
import scipy.sparse as sps
import numpy as np
from dolfin import dx, grad, inner

dolfin.parameters.linear_algebra_backend = "uBLAS"


def ass_convmat_asmatquad():
    N = 2
    mesh = dolfin.UnitSquareMesh(N, N)

    V = dolfin.FunctionSpace(mesh, 'CG', 2)
    W = dolfin.VectorFunctionSpace(mesh, 'CG', 2)

    v = dolfin.TrialFunction(V)
    # vt = dolfin.TestFunction(V)

    w = dolfin.TrialFunction(W)
    wt = dolfin.TestFunction(W)

    NV = V.dim()
    nlist = []
    for k in range(V.dim()):
        # iterate for the rows
        ki = dolfin.Function(V)
        kvec = np.zeros((V.dim(), ))
        kvec[k] = 1
        ki.vector()[:] = kvec
        nklist, nyklist = [], []
        for i in range(V.dim()):
            # iterate for the columns
            bi = dolfin.Function(V)
            bvec = np.zeros((V.dim(), ))
            bvec[i] = 1
            bi.vector()[:] = bvec

            nxik = dolfin.assemble(v * bi.dx(0) * ki * dx)
            nyik = dolfin.assemble(v * bi.dx(1) * ki * dx)

            nklist.extend([sps.csr_matrix(nxik.data()),
                           sps.csr_matrix(nyik.data())])

        nk = sps.hstack(nklist)

        nlist.extend([sps.hstack([nk, sps.csc_matrix((1, 2 * NV ** 2))]),
                      sps.hstack([sps.csc_matrix((1, 2 * NV ** 2)), nk])])

    hmat = sps.vstack(nlist, format='csc')

    # xexp = 'x[0]*x[1]'
    # yexp = 'x[0]*x[1]*x[1]'

    xexp = 'x[0]*x[0]*x[1]'
    yexp = 'x[0]*x[1]*x[1]'

    f = dolfin.Expression((xexp, yexp))
    fx = dolfin.Expression(xexp)
    fy = dolfin.Expression(yexp)

    u = dolfin.interpolate(f, W)
    ux = dolfin.interpolate(fx, V)
    uy = dolfin.interpolate(fy, V)

    # Assemble the 'actual' form
    nform = dolfin.assemble(inner(grad(w) * u, wt) * dx)
    rows, cols, values = nform.data()
    nmat = sps.csr_matrix((values, cols, rows))

    uvec = np.atleast_2d(u.vector().array()).T
    uvecx = np.atleast_2d(ux.vector().array()).T
    uvecy = np.atleast_2d(uy.vector().array()).T
    uvecxy = np.vstack([uvecx, uvecy])

    print np.linalg.norm(nmat * uvec)
    print np.linalg.norm(hmat * np.kron(uvecxy, uvecxy))
    # print np.linalg.norm(nmat*uvec - hmat*np.kron(uvecxy, uvecxy))

    wxinds = W.sub(0).dofmap().dofs()
    wyinds = W.sub(1).dofmap().dofs()

    shifwinds = np.r_[wxinds, wyinds]

    hmat_rowswpdt = col_columns_atend(hmat.T, wyinds)
    nmat_rowswpdt = col_columns_atend(nmat.T, wyinds)

    print np.linalg.norm(hmat_rowswpdt.T*np.kron(uvecxy, uvecxy))

    print np.c_[hmat*np.kron(uvecxy, uvecxy),
                nmat_rowswpdt.T*uvec,
                nmat*uvec,
                hmat_rowswpdt.T*np.kron(uvecxy, uvecxy)]

    print np.linalg.norm(uvec[shifwinds] - uvecxy)


def col_columns_atend(SparMat, ColInd):
    """Shifts a set of columns of a sparse matrix to the right end.

    It takes a sparse matrix and a vector containing indices
    of columns which are appended at the right end
    of the matrix while the remaining columns are shifted to left.

    """

    mat_csr = sps.csr_matrix(SparMat, copy=True)
    MatWid = mat_csr.shape[1]

    # ColInd should not be altered
    ColIndC = np.copy(ColInd)

    for i in range(len(ColInd)):
        subind = ColIndC[i]

        # filter out the current column
        idx = np.where(mat_csr.indices == subind)

        # shift all columns that were on the right by one to the left
        idxp = np.where(mat_csr.indices >= subind)
        mat_csr.indices[idxp] -= 1

        # and adjust the ColInds for the replacement
        idsp = np.where(ColIndC >= subind)
        ColIndC[idsp] -= 1

        # append THE column at the end
        mat_csr.indices[idx] = MatWid - 1

    return mat_csr


if __name__ == '__main__':
    ass_convmat_asmatquad()
