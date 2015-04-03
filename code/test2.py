__author__ = 'ziang'
from util import load_csr_matrix_from_npz
import numpy
import time

if __name__ == "__main__":
    print 'loading x_tr...'
    t0 = time.time()
    x_tr = load_csr_matrix_from_npz('../data/processed/tf_idf_transformation/train/matrix.npz')
    print 'loading finished, time = {0}'.format(time.time()-t0)

    print 'loading y_tr...'
    t0 = time.time()
    y_tr = numpy.loadtxt('../data/processed/tf_idf_transformation/train/labels.csv', dtype='int')
    print 'loading finished, time = {0}'.format(time.time()-t0)
 
    print 'running TruncatedSVD...'
    t0 = time.time()
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100)
    x_tr_new = svd.fit_transform(x_tr, y_tr)
    print 'running TruncatedSVD finished, x_new.shape = {0}, time = {1}'.format(x_tr_new.shape, time.time()-t0)
  
    #delete x_tr
    del x_tr

    print 'fitting model...'
    t0 = time.time()
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_tr_new, y_tr)
    print 'fitting finished, time = {0}'.format(time.time()-t0)

    #delete x_tr_new, y_tr
    del x_tr_new
    del y_tr
  
    print 'loading x_te ...'
    t0 = time.time()
    x_te = load_csr_matrix_from_npz('../data/processed/tf_idf_transformation/test/matrix.npz');
    print 'loading finished, time = {0}'.format(time.time()-t0)

    print 'PCA x_te...'
    t0 = time.time()
    x_te_new = svd.transform(x_te);
    print 'PCA finished, time = {0}'.format(time.time()-t0)

    print 'predicting y...'
    t0 = time.time()
    y_pd = clf.predict_proba(x_te_new)
    print 'predicting finished, time = {0}'.format(time.time()-t0)
  
    print 'saving prediction...'
    t0 = time.time()
    numpy.savetxt('../output/100PCALogistic.csv', y_pd, fmt='%.16f', delimiter=',')
    print 'prediction saved, time = {0}'.format(time.time()-t0)
