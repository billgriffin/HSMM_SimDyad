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

#from utils.cvarray import *
from gaussian_prior import *
import Image


def saveGraph(fn, red, green = None, blue = None):
  """Given a filename and an array of floating point values or None in each of red,  
  green and blue this plots a graph and saves it to the given filename, with 
  automatic scaling. The lengths of red,green and blue should be identical, and 
  they determine the number of pixels in both directions."""
  lines = []
  if red!=None: lines.append((red,numpy.array([0.0,0.0,255.0])))
  if green!=None: lines.append((green,numpy.array([0.0,255.0,0.0])))
  if blue!=None: lines.append((blue,numpy.array([255.0,0.0,0.0])))
  size = len(lines[0][0])

  maxValue = 0.0
  for line in lines: maxValue = max((max(line[0]),maxValue))

  img = numpy.zeros((size,size,3),dtype=numpy.uint8)

  for line in lines:
    for x,point in enumerate(line[0]):
      y = int((size*point)/maxValue)
      if y<0: y = 0
      elif y>=size: y = size-1
      y = size-1-y
      img[y,x,:] += line[1]

  #img = array2cv(img)
  #cv.SaveImage(fn,img)
  img = Image.fromarray(numpy.array(img))
  img.save(fn)


#fname = "../matHigh/c200.txt"
#fname = "../matMed/c219.txt"
fname = "../matLow/c224.txt"
m_data = numpy.transpose(numpy.loadtxt(fname, usecols=(0,),unpack=True))
f_data = numpy.transpose(numpy.loadtxt(fname, usecols=(1,),unpack=True))


# Setup the inital prior...
gp = GaussianPrior(1)
gp.addPrior(numpy.array([0.0]),numpy.array([[0.0]]))

### add real data
for sample in m_data:
  gp.addSample(sample)
  
# Draw a Gaussian to use as ground truth...
gt = gp.sample()
#print 'gt mean =', gt.getMean()[0]
#print 'gt standard deviation =', math.sqrt(gt.getCovariance()[0,0])

# Draw a lot of sample from the ground truth...
samples = map(lambda _: gt.sample(),xrange(32768))

print gp.getMu()


# Create an output directory...
base = 'wag_test_1d'
try: shutil.rmtree(base)
except: pass
os.mkdir(base)

# Select a range to render to the graph...
width = 400
low = gt.getMean()[0] - 3.0*math.sqrt(gt.getCovariance()[0,0])
high = gt.getMean()[0] + 3.0*math.sqrt(gt.getCovariance()[0,0])

# Go through and, starting with no data, increase the amount of data in steps and save an image for each step - green for ground truth, red for the integrated out curve and blue for a draw from the prior...
def render(index, ex=''):
  draw = gp.sample()
  intOut = gp.intProb()
  
  red = []
  green = []
  blue = []
  for i in xrange(width):
    x = float(i)/float(width-1) * (high-low) + low
    red.append(intOut.prob(x))
    green.append(gt.prob(x))
    blue.append(draw.prob(x))
  saveGraph('%s/graph_%06d%s.png'%(base,index,ex),red,green,blue)


render(0)
gp.addSample(samples[0])
render(1)
gp.addSample(samples[1])
render(2)
gp.addSample(samples[2])
gp.addSample(samples[3])
render(4)

scale = 4
while True:
  start = scale
  scale *= 2
  if scale>len(samples): break

  gp.addSamples(samples[start:scale])
  render(scale)

print gp.getMu()
print gp.getLambda()

#gp = GaussianPrior(1)
#gp.addPrior(numpy.array([0.0]),numpy.array([100.0]))
#gp.addSamples(samples)
#render(len(samples),'b')
