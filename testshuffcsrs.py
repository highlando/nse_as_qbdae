import scipy.sparse as sps
import numpy as np


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

xmat = np.arange(12).reshape(3, 4)
ymat = np.arange(12, 24).reshape(3, 4)

sxmat = sps.csr_matrix(xmat)
symat = sps.csr_matrix(ymat)
print sxmat.todense()
print symat.todense()

sxymat = _shuff_mrg_csrmats(sxmat, symat)
print sxymat.todense()

sxyzmat = _pad_csrmats_wzerorows(sxymat.copy(), wheretoput='before')
print sxyzmat.todense()

szxymat = _pad_csrmats_wzerorows(sxymat.copy(), wheretoput='after')

print szxymat.todense()
