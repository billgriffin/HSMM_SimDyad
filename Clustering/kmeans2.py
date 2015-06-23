# File: kmeans2.py
# Desc: make an example of how to cluster objects using K-means algorithm
# use for multiple dimentional data
# first modification: read the inputs from a file

from math import *
from random import randrange
from string import *
import numpy as np

# get list of objects from a file
def openfile ():

   # get the file name:
   print "**** Clustering program: ***** "
   filename = raw_input ("Get the data from file : ")
   openf = open (filename, "r")

   # read from the file, data_source[i]= a string of a line in filename 
   data_source = openf.readlines()
   openf.close() 
   # data is a list of a list of values
   data = []

   for i in range (len(data_source)):
      data.insert(i,[])
      # convert the string into number
      k = 0
      for numStr in split(data_source[i]):
         data[i].insert(k, eval(numStr))
         k +=1

   print "Data converted from file: "
   print data

   return(data)

# function to do k-means clustering alrgorithm
def cluster (data):
   n_point = len(data)
   # let the user decide the number of clusters:
   #n_cluster = input ("How many clusters to group? ")
   n_cluster = 3
   print "n_cluster is = ", n_cluster

   # decide the centroids by randomly picking up an index number
   # and pick the centroid from the data list with the random index
   ran_centroid = []
   print "Making random centroids: "
   for i in range (n_cluster):
      # make sure this random index number is different from other
      test = 0
      first = 0
      print "ran_centroid[",i,"]"
      while (first ==0 or test ==1):
         test = 0
         print "first =", first, " test =", test, 
         ran_num = randrange (0, n_point)
         print " ran_num =", ran_num
         for j in range (i):
            print "ran_centroid [", j,"]", ran_centroid[j]
            if (ran_num == ran_centroid[j]):
               test = 1
         first += 1

      ran_centroid.insert(i,ran_num)
      print "Random centroid is : ", data[ran_centroid[i]]

   # assign points to centroids by:
   # calculating distance from a point to centroids
   # assign a point to the nearest centroid
   data_group = []
   for i in range (n_point):
      min_dis = dis (data[i], data[ran_centroid[0]])

      for j in range (n_cluster):
         dist = dis (data[i], data[ran_centroid[j]])

         if (dist <= min_dis):
            min_dis = dist
            data_group.insert (i,j)
      print "data_group[", i, "]: ", data_group[i]

   # step2:
   # recalculate the centroids
   lst = []
   centroid = []
   for i in range (n_cluster):
      sum = 0
      num_item = 0
      print "Group of centroid[",i,"]: ","centroid= ", data[ran_centroid[i]]

      for j in range (n_point):
         if (data_group[j]== i):
            if (i==0 or len(lst) < num_item+1):
               lst.insert(num_item, data[j])
            else:
               lst[num_item] = data[j]
            num_item += 1
            print "value: ",data[j], " of group: ", data_group[j]

      centroid_valu=[]

      for k in range (len(lst[0])):
         sum = 0
         for h in range (len(lst)):
            print " h= ", h
            sum += lst[h][k]
         centroid_valu.insert(k, sum/len(lst))

      centroid.insert(i, centroid_valu) 
      # now centroid[i] no longer stores index value of the corresponding
      # item in data list, but the real value of the centroid
      print "New centroid value : ", centroid[i]

   # step 3: reassign until no change is made
   count = 1
   print "while loop here: "
   while (count ==1):
      print "   inside while: "
      # making/assigning points to group
      for i in range (n_point):
         min_dis = dis (data[i], centroid[0])

         for j in range (n_cluster):
            dist = dis (data[i], centroid[j])
            if (dist <= min_dis):
               min_dis = dist
               data_group.insert (i,j)
         print "data_group[", i, "]: ", data_group[i]
         #print data_group[i]
      # recalculate the centroids: centroid is the center,
      # calculate: each cordinate of it = average of all vertices' cordinates
      for i in range (n_cluster):
         sum = 0
         num_item = 0
         twodecimals = ['%1.2f' % v for v in centroid[i]]
         print "Group of centroid[",i,"]: ","centroid= ",twodecimals #centroid[i] 
         for j in range (n_point):
            if (data_group[j]== i):
               if (len(lst) < num_item+1):
                  lst.insert(num_item, data[j])
               else:
                  lst[num_item]=data[j]
               num_item +=1
               shortdecimals = ['%1.2f' % v for v in data[j]]
               #print "value: ",data[j], " of group: ", data_group[j]
               print "value: ",shortdecimals, " of group: ", data_group[j]
         centroid_valu =[]
         for k in range (len(lst[0])):
            sum = 0     
            for h in range (len(lst)):
               sum += lst[h][k]
            centroid_valu.insert(k, sum/len(lst))                 
         centroid_val = centroid_valu

         if (centroid_val== centroid[i]):
            count = 0
         else:
            centroid[i] = centroid_val
            shortcentroids = ['%1.2f' % v for v in centroid[i]]
            print "New centroid value : ", shortcentroids #centroid[i]

   # print out the final results
   print
   print "*********Final clustering**********"
   for i in range (n_cluster):
      #print 'Group of centroid[,%.1f,]: ,centroid= %1.3f' % (i,centroid[i])
      twodecimals = ['%1.2f' % v for v in centroid[i]]
      print 'Group of centroid: %s: ,centroid= %s' % (i,twodecimals)
      for j in range (n_point):
         if (i == data_group[j]):
            shortdecimals = ['%1.2f' % v for v in data[j]]
            print "Value: ", shortdecimals, "of group: ", data_group[j]
            #print "Value: ", data[j], "of group: ", data_group[j]

# calculate the distance between 2 points in multi dimentional space
# given 2 points: each point= 1D list of cordinates
def dis (a,b):
   # dimenion of the data
   dim = len(a)
   # calculate the distance in Euclidean metrics
   # can be upgraded to Minkowski metrics for more complex data
   dis = 0
   for i in range (dim):
      dis += (abs(a[i]-b[i]))**2
   dis = sqrt(dis)
   return (dis)

# find the centroid of a group of points
# given a list of all point: 2D list
# do not use here because cannot pass a list to a function
def centroid (lst):
   # centroid value has n-cordinates
   centroid_valu=[]

   for i in range (len(lst[0])):
      sum = 0

      for j in range (len(lst)):
         sum += lst[j][i]

      centroid_valu.insert(i, sum/len(lst))

   return (centroid_valu)    

def main():
   coupleValuesEntire = np.array([
      [1.513,0.842,1.404,1.152,0.032,0.5,0.083,0.511,0.928,2.557,2.5,0.032,0.083,0.965,0.022,1.458,0.769,30.5,8,0.477],
      [3.41,3.3,4.851,1.854,0.789,0.86,5.11,0.683,1.423,1.505,1.896,0.789,5.11,1.868,-0.231,4.131,2.056,2474.5,459,0.674],
      [5.088,1.977,5.932,1.829,3.317,0.749,5.381,0.809,1.166,1.186,1.097,3.317,5.381,0.484,0.078,5.51,1.725,3713.5,625,0.743],
      [4.383,1.764,4.253,1.457,1.568,0.702,2.313,0.626,0.97,1.02,1.144,1.568,2.313,0.389,-0.114,4.318,1.125,2772.5,593,0.584],
      [3.881,1.894,3.931,1.074,1.651,0.64,3.451,0.547,1.013,1.064,1.245,1.651,3.451,0.737,-0.157,3.906,1.138,2235.5,494,0.566],
      [3.96,0.746,3.322,1.193,5.477,0.524,1.297,0.521,0.839,0.664,0.668,5.477,1.297,-1.44,-0.006,3.641,0.755,2128,532,0.5],
      [2.082,2.061,3.116,2.319,0.364,0.633,0.58,0.728,1.496,1.58,1.375,0.364,0.58,0.465,0.139,2.599,1.861,1168.5,249,0.587],
      [1.764,1.286,1.158,1.275,0.184,0.525,0.079,0.5,0.657,0.451,0.473,0.184,0.079,-0.841,-0.048,1.461,0.821,16,4,0.5],
      [4.251,2.681,3.072,1.885,1.133,0.812,0.853,0.593,0.723,0.633,0.867,1.133,0.853,-0.283,-0.314,3.662,1.593,1976,410,0.602],
      [4.513,2.182,3.85,2.182,1.831,0.734,1.102,0.7,0.853,0.774,0.811,1.831,1.102,-0.507,-0.047,4.181,1.75,2457.5,477,0.644],
      [0.767,0.649,2.651,0.98,0.018,0.5,0.442,0.501,3.458,17.019,17,0.018,0.442,3.181,0.001,1.709,0.676,59,15,0.492],
      [4.021,1.147,3.207,1.678,0.97,0.636,0.474,0.63,0.798,0.648,0.654,0.97,0.474,-0.715,-0.008,3.614,1.106,1848,429,0.538],
      [4.949,2.565,5.018,1.178,2.248,0.811,21.531,0.639,1.014,1.087,1.381,2.248,21.531,2.26,-0.239,4.983,1.539,3255,592,0.687],
      [3.225,1.667,3.322,1.983,0.571,0.642,0.7,0.671,1.03,1.184,1.134,0.571,0.7,0.205,0.044,3.274,1.28,1558.5,367,0.531],
      [4.37,0.761,4.819,1.03,12.315,0.561,12.073,0.621,1.103,1.106,0.998,12.315,12.073,-0.02,0.103,4.595,0.793,3231.5,695,0.581],
      [4.755,1.176,4.818,1.248,5.145,0.637,6.815,0.635,1.013,1.038,1.042,5.145,6.815,0.281,-0.004,4.786,0.932,3330.5,682,0.61],
      [5.715,1.287,5.33,2.644,43.812,0.722,1.525,0.923,0.933,0.79,0.618,43.812,1.525,-3.358,0.246,5.522,1.901,3571,587,0.76],
      [3.668,1.238,4.836,1.811,1.495,0.561,2.716,0.714,1.318,1.552,1.22,1.495,2.716,0.597,0.241,4.252,1.322,2730,575,0.593],
      [5.668,2.612,5.841,1.304,2.81,0.876,13.733,0.755,1.031,1.09,1.264,2.81,13.733,1.587,-0.148,5.755,1.504,3691.5,618,0.747],
      [4.602,1.466,5.358,2.597,3.017,0.653,1.959,0.883,1.164,1.192,0.881,3.017,1.959,-0.432,0.301,4.98,1.591,3274.5,594,0.689],
      [4.237,1.212,3.793,1.384,1.548,0.631,1.267,0.613,0.895,0.895,0.92,1.548,1.267,-0.2,-0.028,4.015,1.106,2218,473,0.586],
      [4.507,0.833,3.714,0.906,8.244,0.587,0.729,0.586,0.824,0.472,0.473,8.244,0.729,-2.425,-0.001,4.11,0.604,2793.5,664,0.526],
      [4.297,0.741,6.95,1.054,13.979,0.548,64.364,0.877,1.618,1.686,1.055,13.979,64.364,1.527,0.469,5.623,0.665,4029.5,715,0.704],
      [5.929,2.135,6.54,1.296,4.15,0.844,54.462,0.825,1.103,1.191,1.219,4.15,54.462,2.574,-0.023,6.235,1.461,4438,703,0.789],
      [4.429,0.832,6.222,1.186,14.021,0.566,59.083,0.784,1.405,1.458,1.053,14.021,59.083,1.438,0.325,5.326,0.85,3825.5,717,0.667],
      [5.457,1.425,4.369,1.509,14.298,0.706,2.44,0.639,0.801,0.687,0.759,14.298,2.44,-1.768,-0.1,4.913,1.082,3444,684,0.629],
      [6.093,0.895,5.774,1.804,718,0.762,6.49,0.781,0.948,0.89,0.868,718,6.49,-4.706,0.025,5.934,1.095,4257.5,717,0.742],
      [3.389,0.751,7.305,1.166,0.97,0.506,718,0.913,2.156,3.658,2.028,0.97,718,6.607,0.59,5.347,0.728,3839,718,0.668],
      [3.885,0.802,4.568,1.72,2.587,0.537,2.042,0.694,1.176,1.203,0.931,2.587,2.042,-0.236,0.257,4.226,0.971,2602,567,0.574],
      [4.29,1.536,5.106,2.39,1.899,0.646,2.784,0.782,1.19,1.36,1.123,1.899,2.784,0.383,0.191,4.698,1.481,3011.5,579,0.65]])    
   #data = openfile ()
   from numpy import genfromtxt
   #firstHalf = genfromtxt('FirstHalfCouples.csv', delimiter=',')
   secondHalf = genfromtxt('secondHalfCouples.csv', delimiter=',')
   #print secondHalf
   coupleValuesEntire = secondHalf
   #### to reduce use below
   values = []
   ###for i in coupleValuesEntire:
   for j in range(len(coupleValuesEntire)):
      values.append([coupleValuesEntire[j][0],coupleValuesEntire[j][1],coupleValuesEntire[j][2],coupleValuesEntire[j][3],\
                     coupleValuesEntire[j][8],coupleValuesEntire[j][10]])  
   data = np.array(values)
   #print data
   #data = coupleValuesEntire
   data = cluster(data)

main()


