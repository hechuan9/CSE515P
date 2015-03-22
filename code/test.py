__author__ = 'ziang'
from util import load_csr_matrix_from_npz
import numpy
import time

if __name__ == "__main__":
    print 'loading x...'
    t0 = time.time()
    x_tr = load_csr_matrix_from_npz('../data/processed/tf_idf_transformation/train/matrix.npz')
    print 'loading finished, time = {0}'.format(time.time()-t0)
    print 'loading y...'
    t0 = time.time()
    y_tr = numpy.loadtxt('../data/processed/tf_idf_transformation/train/labels.csv', dtype='int')
    print 'loading finished, time = {0}'.format(time.time()-t0)
 
    print 'running PCA...'
    t0 = time.time()
    from sklearn.decomposition import RandomizedPCA
    selector = RandomizedPCA(n_components=100)
    x_tr_new = selector.fit_transform(x_tr, y_tr)
    print 'running PCA finished, x_new.shape = {0}, time = {1}'.format(x_tr_new.shape, time.time()-t0)
  
  
    print 'fitting model...'
    t0 = time.time()
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_tr_new[0:x_tr_new.shape[0]/2, :], y_tr[0:len(y_tr)/2])
    print 'fitting finished, time = {0}'.format(time.time()-t0)
  
    print 'predicting y...'
    t0 = time.time()
    y_pd = clf.predict(x_tr_new[x_tr_new.shape[0]/2:, :])
    print 'predicting finished, time = {0}'.format(time.time()-t0)
  
    print 'computing error...'
    y = y_tr[len(y_tr)/2: len(y_tr)]
    count = 0
    for i in xrange(0, len(y)):
        if y[i] != y_pd[i]:
            count += 1
    print 'Error = {0}'.format(count / len(y))