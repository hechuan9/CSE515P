__author__ = 'ziang'


def save_data_to_npz(file_name, x):
    from numpy import savez
    savez(file_name, data=x.data, indices=x.indices, indptr=x.indptr, shape=x.shape)


def load_csr_matrix_from_npz(file_name):
    from numpy import load
    from scipy.sparse import csr_matrix
    loader = load(file_name)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def save_id_to_csv(file_name, ids):
    from csv import writer
    ids_dict = {}
    for i in xrange(0, len(ids)):
        ids_dict[i] = ids[i]
    w = writer(open(file_name, "w"))
    for key, val in ids_dict.items():
        w.writerow([key, val])


def load_id_from_csv(file_name):
    from csv import reader
    ids = {}
    for key, val in reader(open(file_name)):
        ids[key] = val
    return ids
