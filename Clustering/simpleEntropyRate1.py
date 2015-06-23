'''entropy rate for chain that generates the matrix'''
import numpy as np
import time


m = np.array([
   [0.11428571428571,0.05714285714286,0.02857142857143,0.02857142857143],
   [0.05714285714286,0.11428571428571,0.02857142857143,0.02857142857143],
   [0.05714285714286,0.05714285714286,0.05714285714286,0.05714285714286],
   [0.22857142857143,0.02857142857143,0.02857142857143,0.02857142857143]])

margins = np.array([.22857, .22857,.22857,.31429])  # marginal prob

entropies = []

def entropy_rate():
   for i in range(len(margins)):
      e = 0.
      for j in range(len(m[i])):
         ''' cell entropy * marginal value '''
         e += (-m[i][j] * np.log2(m[i][j]))* margins[i]
      entropies.append(e)
   return np.sum(entropies)

print entropy_rate()


entropies2 = []
def entropy_rate2():
   for i in range(len(margins)):
      ee = 0.
      for x in np.nditer(m[i], op_flags=['readwrite']):
         x[...] = (-x * np.log2(x)) * margins[i]
      entropies2.append(m[i])
   return np.sum(entropies2)

### about the same speed
print entropy_rate2()  #note: this changes the original matrix