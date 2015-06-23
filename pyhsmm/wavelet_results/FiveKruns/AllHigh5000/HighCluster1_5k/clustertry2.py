"""
==================
GMM classification
==================

Demonstration of Gaussian mixture models for classification.

See :ref:`gmm` for more information on the estimator.

Plots predicted labels on both training and held out test data using a
variety of GMM classifiers on the iris dataset.

Compares GMMs with spherical, diagonal, full, and tied covariance
matrices in increasing order of performance.  Although one would
expect full covariance to perform best in general, it is prone to
overfitting on small datasets and does not generalize well to held out
test data.

On the plots, train data is shown as dots, while test data is shown as
crosses. The iris dataset is four-dimensional. Only the first two
dimensions are shown here, and thus some points are separated in other
dimensions.
"""
print(__doc__)

# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# License: BSD 3 clause

# $Id$

import pylab as pl
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
#from sklearn.externals.six.moves import xrange

from sklearn.mixture import GMM


def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


iris = [[8.0,8.0,15,90,22.4008378422],
        [8.0,8.0,20,60,25.2343594579],
        [10.0,8.0,25,60,25.6829557058],
        [10.0,10.0,15,30,26.3230279732],
        [8.0,10.0,20,90,26.8309739422],
        [8.0,10.0,15,10,27.7811668691],
        [10.0,8.0,15,10,28.4787107887],
        [8.0,10.0,15,60,28.8044597892],
        [8.0,8.0,25,90,28.9396691924],
        [8.0,8.0,25,30,28.9695153018],
        [8.0,10.0,20,30,28.9917787393],
        [8.0,10.0,25,90,29.3386636405],
        [8.0,10.0,20,60,29.5693532596],
        [8.0,10.0,25,10,30.0181270455],
        [8.0,8.0,20,30,30.2030315636],
        [10.0,10.0,20,30,30.5495686888],
        [8.0,10.0,20,10,30.6853334208],
        [10.0,8.0,20,30,30.8367759548],
        [8.0,8.0,20,10,30.8879572909],
        [10.0,8.0,20,60,30.9217899424],
        [10.0,10.0,25,60,30.9474424318],
        [10.0,10.0,15,90,31.3581787391],
        [8.0,8.0,20,90,31.5419781773],
        [10.0,10.0,25,30,31.6490087561],
        [10.0,8.0,15,60,31.7147473609],
        [10.0,8.0,25,90,32.3917624029],
        [10.0,8.0,15,90,32.659017816],
        [10.0,8.0,15,30,32.8371656214],
        [8.0,8.0,15,60,33.1244535603],
        [8.0,8.0,15,30,33.3494159934],
        [8.0,8.0,15,10,33.3761112299],
        [10.0,10.0,20,90,33.7790862698],
        [10.0,10.0,20,60,33.9753786497],
        [10.0,10.0,25,10,34.0490181734],
        [8.0,8.0,25,60,34.8702516154],
        [10.0,8.0,25,10,35.0451389491],
        [8.0,10.0,15,30,35.0820375393],
        [10.0,8.0,20,90,35.4618943388],
        [10.0,10.0,15,60,35.5669791409],
        [10.0,8.0,25,30,35.6169425313],
        [8.0,8.0,25,10,35.8043555547],
        [10.0,10.0,20,10,35.8099566759],
        [8.0,10.0,25,60,36.2083388313],
        [8.0,10.0,25,30,37.0995008261],
        [10.0,10.0,15,10,37.3915136467],
        [10.0,8.0,20,10,39.8884178163],
        [8.0,10.0,15,90,41.7073641969]]
        
#iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(iris.target, n_folds=4)
# Only take the first fold.
train_index, test_index = next(iter(skf))


X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

pl.figure(figsize=(3 * n_classifiers / 2, 6))
pl.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                   left=.01, right=.99)


for index, (name, classifier) in enumerate(classifiers.iteritems()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    h = pl.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)

    for n, color in enumerate('rgb'):
        data = iris.data[iris.target == n]
        pl.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                   label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate('rgb'):
        data = X_test[y_test == n]
        pl.plot(data[:, 0], data[:, 1], 'x', color=color)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    pl.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
            transform=h.transAxes)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    pl.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
            transform=h.transAxes)

    pl.xticks(())
    pl.yticks(())
    pl.title(name)

pl.legend(loc='lower right', prop=dict(size=12))


pl.show()
