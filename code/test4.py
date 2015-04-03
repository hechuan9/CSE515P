__author__ = 'chuan'
from util import load_csr_matrix_from_npz
import tables as tb
from numpy import array
from scipy import sparse
import time

if __name__ == "__main__":
    print 'loading x_tr...'
    t0 = time.time()
    M = load_csr_matrix_from_npz('../data/processed/tf_idf_transformation/train/matrix.npz')
    print 'loading finished, time = {0}'.format(time.time()-t0)

    store_sparse_mat(M, 'M')

import tables as tb
from numpy import array
from scipy import sparse

def store_sparse_mat(m, name, store='store.h5'):
    msg = "This code only works for csr matrices"
    assert(m.__class__ == sparse.csr.csr_matrix), msg
    with tb.openFile(store,'a') as f:
        for par in ('data', 'indices', 'indptr', 'shape'):
            full_name = '%s_%s' % (name, par)
            try:
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass

            arr = array(getattr(m, par))
            atom = tb.Atom.from_dtype(arr.dtype)
            ds = f.createCArray(f.root, full_name, atom, arr.shape)
            ds[:] = arr