#!/usr/bin/env python
def main():
    """Calculate best fit line through vectors and plot"""
    from mayavi.mlab import points3d,savefig,show,outline,plot3d
    from numpy import array,sum,transpose
    from numpy.linalg import eig
    dat=open('vecs.xyz','rU').readlines()
    x,y,z=[],[],[]
    for line in dat:
        rec=line.split()
        x.append(float(rec[0]))
        y.append(float(rec[1]))
        z.append(float(rec[2]))
    X,Y,Z=array(x),array(y),array(z)
    points3d(X,Y,Z,color=(0,0,0),scale_factor=0.25,opacity=.5)
    outline(color=(.7,0,0))
    T=array([[sum(X*X),sum(X*Y),sum(X*Z)],\
         [sum(Y*X),sum(Y*Y),sum(Y*Z)],\
          [sum(Z*X),sum(Z*Y),sum(Z*Z)]])
    evals,evects=eig(T)
    pv=transpose(evects)[0]*3.
    plot3d([pv[0],-pv[0]],[pv[1],-pv[1]],[pv[2],-pv[2]],tube_radius=0.1,color=(0,1,0))
    show()
main()
