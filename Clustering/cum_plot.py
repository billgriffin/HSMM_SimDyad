import numpy as np

if __name__ == '__main__':
   dyad_data = np.loadtxt( 'c220.txt' )
   male = []
   female = []
   dyadicstate = []
   for i in range(len(dyad_data)):
      male.append(int(dyad_data[i][0])) 
      female.append(int(dyad_data[i][1])) 
      dyadicstate.append( (int(dyad_data[i][0]) + int(dyad_data[i][1]) )/2.)        

   male = np.array(male)
   female = np.array(female)
   dyadicstate = np.array(dyadicstate)

   file = male  #[2,3,4,1,5,7,8,4]
   tally = np.array([0,0,0])
   tallys = np.zeros((len(file),3))
   for i in range(len(file)):    
      if file[i] < 4:      
         tally[0] += 1
         tallys[i] = tally      
      elif file[i] == 4:
         tally[1] += 1
         tallys[i] = tally          
      else:
         tally[2] += 1      
         tallys[i] = tally

   #print x

print tallys
for i in tallys: print '%s' % i
#m_cumsum = np.cumsum(tallys, axis=0)
#for i in m_cumsum: print '%s' % i
#print m_cumsum
'''http://stackoverflow.com/questions/20687164/convert-a-3d-
drillhole-trace-to-cartesian-coordinates-and-plot-it-with-matplotlib'''
import matplotlib.pyplot as plt
# import for 3d plot
from mpl_toolkits.mplot3d import Axes3D
# initializing 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
# several data points 
r = tallys[:,1]
#r = m_cumsum[:,1]
# get lengths of the separate segments 
#r[1:] = r[1:] - r[:-1]
phi = tallys[:,0]
theta = tallys[:,2]
#phi = m_cumsum[:,0]
#theta = m_cumsum[:,2]

# convert to radians
phi = phi * 2 * np.pi / 360.
# in spherical coordinates theta is measured from zenith down; you are measuring it from horizontal plane up 
theta = (90. - theta) * 2 * np.pi / 360.
# get x, y, z from known formulae
x = r*np.cos(phi)*np.sin(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*np.cos(theta)
#for i in range(len(file)): print x[i],y[i],z[i]
# np.cumsum is employed to gradually sum resultant vectors 
#ax.plot(np.cumsum(x),np.cumsum(y),np.cumsum(z))
#ax.plot(np.cumsum(phi),np.cumsum(theta),np.cumsum(r))
xs = (len(file),phi)
ys = (len(file), theta)
zs = (len(file), r)
ax.scatter(xs,ys,zs)
plt.show()