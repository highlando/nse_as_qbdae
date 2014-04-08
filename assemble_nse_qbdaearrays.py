import dolfin
import scipy.sparse as sps
import numpy as np
from dolfin import dx, grad, inner

import dolfin_navier_scipy as dns
import dolfin_navier_scipy.dolfin_to_sparrays as dnsts

dolfin.parameters.linear_algebra_backend = "uBLAS"


def test_qbdae_ass():
    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc, \
        data_prfx, ddir, proutdir = \
        dns.problem_setups.get_sysmats(problem='drivencavity', N=25, nu=1e-2)

    invinds = femp['invinds']
    # nv = femp['V'].dim()
    # invinds = np.arange(nv)

    ass_convmat_asmatquad(W=femp['V'], invindsw=invinds)


def ass_convmat_asmatquad(W=None, invindsw=None):
    """ assemble the convection matrix H, so that N(v)v = H[v.v]

    note that the boundary conditions have to be treated properly
    """
    mesh = W.mesh()

    V = dolfin.FunctionSpace(mesh, 'CG', 2)

    # this is very specific for V being a 2D VectorFunctionSpace
    invindsv = invindsw[::2]/2

    v = dolfin.TrialFunction(V)
    vt = dolfin.TestFunction(V)

    w = dolfin.TrialFunction(W)
    wt = dolfin.TestFunction(W)

    def _pad_csrmats_wzerorows(smat, wheretoput='before'):
        """add zero rows before/after each row

        """
        indpeter = smat.indptr
        auxindp = np.c_[indpeter, indpeter].flatten()
        if wheretoput == 'after':
            smat.indptr = auxindp[1:]
        else:
            smat.indptr = auxindp[:-1]

        smat._shape = (2*smat.shape[0], smat.shape[1])

        return smat

    def _shuff_mrg_csrmats(xm, ym):
        """shuffle merge csr mats [xxx],[yyy] -> [xyxyxy]

        """
        xm.indices = 2*xm.indices
        ym.indices = 2*ym.indices + 1
        xm._shape = (xm.shape[0], 2*xm.shape[1])
        ym._shape = (ym.shape[0], 2*ym.shape[1])
        return xm + ym

    nklist = []
    for i in invindsv:
    # for i in range(V.dim()):
        # iterate for the columns
        bi = dolfin.Function(V)
        bvec = np.zeros((V.dim(), ))
        bvec[i] = 1
        bi.vector()[:] = bvec

        nxi = dolfin.assemble(v * bi.dx(0) * vt * dx)
        nyi = dolfin.assemble(v * bi.dx(1) * vt * dx)

        rows, cols, values = nxi.data()
        nxim = sps.csr_matrix((values, cols, rows))

        rows, cols, values = nyi.data()
        nyim = sps.csr_matrix((values, cols, rows))

        nxyim = _shuff_mrg_csrmats(nxim, nyim)
        nxyim = nxyim[invindsv, :][:, invindsw]
        nyxxim = _pad_csrmats_wzerorows(nxyim.copy(), wheretoput='after')
        nyxyim = _pad_csrmats_wzerorows(nxyim.copy(), wheretoput='before')

        nklist.extend([nyxxim, nyxyim])

    hmat = sps.hstack(nklist, format='csc')

    xexp = '(1-x[0])*x[0]*(1-x[1])*x[1]*x[0]*0'
    yexp = '(1-x[0])*x[0]*(1-x[1])*x[1]*x[1]+1'
    # yexp = 'x[0]*x[0]*x[1]*x[1]'

    f = dolfin.Expression((xexp, yexp))

    u = dolfin.interpolate(f, W)
    uvec = np.atleast_2d(u.vector().array()).T

    uvec_gamma = uvec.copy()
    uvec_gamma[invindsw] = 0
    u_gamma = dolfin.Function(W)
    u_gamma.vector().set_local(uvec_gamma)

    uvec_i = 0*uvec
    uvec_i[invindsw, :] = uvec[invindsw]
    u_i = dolfin.Function(W)
    u_i.vector().set_local(uvec_i)

    # Assemble the 'actual' form
    nform = dolfin.assemble(inner(grad(w) * u, wt) * dx)
    rows, cols, values = nform.data()
    nmat = sps.csr_matrix((values, cols, rows))
    nmatrc = nmat[invindsw, :][:, :]
    # nmatrccc = nmatrc[:, :][:, invindsw]
    # nmatc = nmat[invindsw, :][:, :]

    N1, N2, fv = dnsts.get_convmats(u0_dolfun=u_gamma, V=W)

    # print np.linalg.norm(nmatc * uvec[invindsw])
    print np.linalg.norm(nmatrc * uvec)
    print np.linalg.norm(nmatrc * uvec_i + nmatrc * uvec_gamma)
    print np.linalg.norm(hmat * np.kron(uvec[invindsw], uvec[invindsw])
                         + ((N1+N2)*uvec_i)[invindsw, :] + fv[invindsw, :])
    # print np.linalg.norm((hmat * np.kron(uvec, uvec))[invindsw, :])
    # print np.linalg.norm(((N1+N2)*uvec)[invindsw, :] + fv[invindsw, :])
    print 'consistency tests'
    print np.linalg.norm(uvec[invindsw]) - np.linalg.norm(uvec_i)
    print np.linalg.norm(uvec - uvec_gamma - uvec_i)


if __name__ == '__main__':
    test_qbdae_ass()
