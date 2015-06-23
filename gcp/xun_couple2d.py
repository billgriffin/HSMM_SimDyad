#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import os
import shutil
import math
import numpy
#import cv
import Image
#from utils.cvarray import *
from gaussian_prior import *




# Create an output directory...
base = 'couple'
try: shutil.rmtree(base)
except: pass
os.mkdir(base)

# Select a range to render to the graph...
width = 400
height = 400
lowX = 0
highX = 10
lowY = 0
highY = 10



# Go through and, starting with no data, increase the amount of data in steps and save an image for each step - green for ground truth, red for the integrated out curve and blue for a draw from the prior...
def saveGraph(fn, red, green = None, blue = None):
  """Takes 3 optional 2d arrays of floats, normalises them and saves them to the 3 channels of an image."""
  maxValue = 0.0
  if red!=None: maxValue = max((maxValue,red.max()))
  if green!=None: maxValue = max((maxValue,green.max()))
  if blue!=None: maxValue = max((maxValue,blue.max()))

  img = numpy.zeros((width,height,3),dtype=numpy.uint8)

  if red!=None: img[:,:,2] = red * (255.0/maxValue)
  if green!=None: img[:,:,1] = green * (255.0/maxValue)
  if blue!=None: img[:,:,0] = blue * (255.0/maxValue)

  img = Image.fromarray(img) 
  img.save(fn)


def render(index, ex=''):
  draw = gp.sample()
  intOut = gp.intProb()

  red = numpy.zeros((width,height),dtype=numpy.float32)
  green = numpy.zeros((width,height),dtype=numpy.float32)
  blue = numpy.zeros((width,height),dtype=numpy.float32)
  for yPos in xrange(height):
    y = float(yPos)/float(height-1) * (highY-lowY) + lowY
    for xPos in xrange(width):
      x = float(xPos)/float(width-1) * (highX-lowX) + lowX
      red[yPos,xPos] = intOut.prob([x,y])
      green[yPos,xPos] = 0 #gt.prob([x,y]) #was 0
      blue[yPos,xPos] = draw.prob([x,y])
  saveGraph('%s/graph_%06d%s.png'%(base,index,ex),red,green,blue)


#fname = "../matHigh/c200.txt"
#fname = "../matMed/c219.txt"
fname = "../matLow/c224.txt"
data = numpy.transpose(numpy.loadtxt(fname, usecols=(0,1),unpack=True))

# Setup the inital prior...
gp = GaussianPrior(2)
d_len = len(data)
#gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[d_len,0.0],[0.0,d_len]]))
gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))
########################################################################
## Draw a Gaussian to use as ground truth...
#gt = gp.sample()
#print 'gt mean =', gt.getMean()[0]
#print 'gt standard deviation =', math.sqrt(gt.getCovariance()[0,0])
#######################################################################
print gp.getMu()
print gp.getLambda()
print

for sample in data:
  gp.addSample(sample)
render(0,'b')

print gp.getMu()
print gp.getLambda()
print
print gp.getK()
print gp.getN()

# for hyper-parameters of poisson distribution
#
# get durations for each pair in data
#  e.g. (1,2): 3,5,7,4  mean1
#       (3,2): 2,1,3,2  mean2
#       ....            meanN
# mean1->meanN: Gamma distribution