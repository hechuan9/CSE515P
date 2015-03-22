import os
import libarchive

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time

import util


def extract_hashing_feature_from_compressed_data(input_file, num_entries, output_path):
    hv = HashingVectorizer(encoding='ISO-8859-2')

    count_total = num_entries
    # transform to hashing vector
    print 'Start transform data to hashing vector'
    t0 = time.time()
    count = 0
    with libarchive.file_reader(input_file) as archive:
        for entry in archive:
            count += 1
            if entry.pathname.find('.bytes') != -1:
                text = [" ".join([block.replace('\r\n', ' ') for block in entry.get_blocks()])]
                util.save_data_to_npz(output_path + entry.pathname, hv.transform(text))
            t = time.time() - t0
            print 'Transform:\tfiles: ' + str(count_total - count) + '/' + str(count_total) \
                  + '\tElapsed time: ' + str(t) + '(s)' \
                  + '\tTime left: ' + str((t / count) * (count_total - count)) + '(s)'


def convert_hashing_feature_to_tf_idf(input_path, output_path):
    # training data
    # load data from hashing vector
    print 'Data loading...{0}'.format(input_path + 'train/')
    t0 = time.time()
    file_names = os.listdir(input_path + 'train/')
    x_tr = csr_matrix(load_and_merge_matrices(input_path + 'train/', file_names))
    print 'Loading finished, x_tr.shape={0} time={1}'.format(x_tr.shape, time.time()-t0)

    # transform it into tf-idf form
    print 'Start transforming...'
    t0 = time.time()
    col_indices = sorted(set(x_tr.indices))
    x_tr = transform_to_tf_idf(x_tr[:, col_indices])
    print 'Transforming finished, x_tr.shape={0} time={1}'.format(x_tr.shape, time.time()-t0)

    # save data
    print 'Saving data...'
    t0 = time.time()
    util.save_data_to_npz(output_path + 'train/matrix', x_tr)
    for i in xrange(0, len(file_names)):
        file_names[i] = file_names[i].replace('.bytes.npz', '')
    util.save_id_to_csv(output_path + 'train/id.csv', file_names)
    print 'Saving finished, time={0}'.format(time.time()-t0)


    # testing data
    # load data from hashing vector
    print 'Data loading...'
    t0 = time.time()
    file_names = os.listdir(input_path + 'test/')
    x_te = csr_matrix(load_and_merge_matrices(input_path + 'test/', file_names))
    print 'Loading finished, x_tr.shape={0} time={1}'.format(x_te.shape, time.time()-t0)

    # transform it into tf-idf form
    print 'Start transforming...'
    t0 = time.time()
    x_te = transform_to_tf_idf(x_te[:, col_indices])
    print 'Transforming finished, x_tr.shape={0} time={1}'.format(x_te.shape, time.time()-t0)

    # save data
    print 'Saving data...'
    t0 = time.time()
    util.save_data_to_npz(output_path + 'test/matrix', x_te)
    for i in xrange(0, len(file_names)):
        file_names[i] = file_names[i].replace('.bytes.npz', '')
    util.save_id_to_csv(output_path + 'test/id.csv', file_names)
    print 'Saving finished, time={0}'.format(time.time()-t0)


def transform_to_tf_idf(x):
    tt = TfidfTransformer()
    return tt.fit_transform(x)


def reorder_training_labels():
    id = util.load_id_from_csv('../data/processed/tf_idf_transformation/train/id.csv')
    lb = util.load_id_from_csv('../data/raw/trainLabels.csv')
    y = []
    for i in xrange(0, len(id)):
        y.append(int(lb[id[str(i)]]))
    # with open('../data/processed/tf_idf_transformation/train/labels.csv', 'w') as f:
    #     f.writelines('\r\n'.join(y))
    print y
    np.savetxt('../data/processed/tf_idf_transformation/train/labels.csv', y, fmt='%d')


def load_and_merge_matrices(path, names):
    length = len(names)
    if length == 1:
        return util.load_csr_matrix_from_npz(path + names[0])
    else:
        return vstack([load_and_merge_matrices(path, names[0:length/2]),
                       load_and_merge_matrices(path, names[length/2:length])])


def test():
    extract_hashing_feature_from_compressed_data('../data/dataSample.7z', 4, '../data/processed/sample/')
    convert_hashing_feature_to_tf_idf('../data/processed/sample/', '../data/processed/sample_out/')
    x = util.load_csr_matrix_from_npz('../data/processed/sample_out/matrix.npz')
    print type(x)
    tt = TfidfTransformer()
    t0 = time.time()
    indices_1 = sorted(set(x.indices))
    indices_2 = sorted(set(np.nonzero(x)[1]))
    for i in xrange(0, 548283):
        if indices_1[i] != indices_2[i]:
            print 'Whoops!'
    print 'same'
    # x_tr = tt.fit_transform(x[:, sorted(set(x.indices))])
    print time.time()-t0


if __name__ == "__main__":
    raw_data_path = '../data/raw/'
    hv_data_path = '../data/processed/hashing_vectorization/'
    ti_data_path = '../data/processed/tf_idf_transformation/'

    # extract_hashing_feature_from_compressed_data(raw_data_path + 'train.7z', 21736, ti_data_path)
    # extract_hashing_feature_from_compressed_data(raw_data_path + 'test.7z', 21746, ti_data_path)

    # convert_hashing_feature_to_tf_idf(hv_data_path, ti_data_path)
    # test()
    reorder_training_labels()


