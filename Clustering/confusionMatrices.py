import numpy as np
matReal = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2])
matCat3Short = np.array([0,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
matCat3Long = np.array([0,0,2,1,0,0,0,0,0,0,0,0,2,0,2,2,2,1,2,2,0,1,2,2,2,2,2,2,1,1])
firstHalfShort = np.array([0,0,2,0,0,0,1,0,0,2,1,0,2,1,2,2,0,2,2,0,0,0,2,2,2,2,2,1,0,2])
firstHalfLong = np.array([0,1,2,1,0,1,0,0,0,2,0,1,2,0,2,2,0,1,2,1,0,1,2,2,2,2,2,2,0,1])
secondHalfShort = np.array([0,1,2,0,1,0,0,0,0,0,1,0,1,0,1,0,2,1,2,2,1,1,2,2,2,2,2,2,2,1])

lamda9 = np.array([0,0,0,0,0,1,1,1,1,2,0,1,1,1,1,2,2,2,2,2,0,0,1,1,1,2,2,2,2,2])

from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(matReal, matCat3Short)
#cm = confusion_matrix(matReal, matCat3Long)
#cm = confusion_matrix(matReal, firstHalfShort)
##cm = confusion_matrix(matReal, firstHalfLong)
#cm = confusion_matrix(matReal, secondHalfShort)

#print cm

import pylab as pl

fig = pl.figure()
alpha = ['High', 'Med', 'Low']
ax = fig.add_subplot(221)
cm = confusion_matrix(matReal, lamda9)
cax = ax.matshow(cm,cmap ='YlOrRd')
fig.colorbar(cax)
pl.title('Confusion matrix: lamda9')
pl.ylabel('True label')
pl.xlabel('Predicted label')
for i in xrange(0,3):
    for j in xrange(0,3):
        pl.text(j,i, "{0:5.2f}".format(cm[i][j]),
            color='k',
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center")

ax.set_xticklabels(['']+alpha)
ax.set_yticklabels(['']+alpha)


ax = fig.add_subplot(222)
cm = confusion_matrix(matReal, matCat3Long)
cax = ax.matshow(cm,cmap ='YlOrRd')
fig.colorbar(cax)
pl.title('Confusion matrix: Entire Long')
pl.ylabel('True label')
pl.xlabel('Predicted label')
for i in xrange(0,3):
    for j in xrange(0,3):
        pl.text(j,i, "{0:5.2f}".format(cm[i][j]),
            color='k',
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center")
ax.set_xticklabels(['']+alpha)
ax.set_yticklabels(['']+alpha)

ax = fig.add_subplot(223)
cm = confusion_matrix(matReal, firstHalfShort)
cax = ax.matshow(cm,cmap ='YlOrRd')
fig.colorbar(cax)
pl.title('Confusion matrix: First Short')
pl.ylabel('True label')
pl.xlabel('Predicted label')
for i in xrange(0,3):
    for j in xrange(0,3):
        pl.text(j,i, "{0:5.2f}".format(cm[i][j]),
            color='k',
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center")
ax.set_xticklabels(['']+alpha)
ax.set_yticklabels(['']+alpha)
#pl.grid()

ax = fig.add_subplot(224)
cm = confusion_matrix(matReal, secondHalfShort)
cax = ax.matshow(cm,cmap ='YlOrRd') #edgecolors='k', linewidths=2)
fig.colorbar(cax)
pl.title('Confusion matrix: Second Short')
pl.ylabel('True label')
pl.xlabel('Predicted label')
for i in xrange(0,3):
    for j in xrange(0,3):
        pl.text(j,i, "{0:5.2f}".format(cm[i][j]),
            color='k',
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center")
ax.set_xticklabels(['']+alpha)
ax.set_yticklabels(['']+alpha)


#pl.matshow(cm)
#pl.title('Confusion matrix')
#pl.colorbar()
#pl.ylabel('True label')
#pl.xlabel('Predicted label')
#pl.grid()

pl.show()
