"""This visualizes the ratios of distances creating by taking
each distance and dividing by the lowest distance """

from pylab import *
A = [[1.12,1.24,1.27,1.00],
     [1.01,1.13,1.23,1.35],
     [1.26,1.23,1.08,1.44]]

D =[[1.49,1.01,1.29,1.00],
    [1.11,1.00,1.13,1.16],
    [1.56,1.11,1.16,1.21]]

figure(1)
imshow(A, interpolation='nearest')
title('All High (10k) Comparison of Distances A:\n Ratio to the Lowest (1.0)')
xticks(arange(4),('10','30','60','90'))
xlabel('Truncation Level')
yticks(arange(3),('15','20','25'))
ylabel('States')
cbar = colorbar()
cbar.ax.set_ylabel('Distance Ratio')
grid(True)

figure(2)
imshow(D, interpolation='nearest')
title('All High (10k) Comparison of Distances D:\n Ratio to the Lowest (1.0)')
xticks(arange(4),('10','30','60','90'))
xlabel('Truncation Level')
yticks(arange(3),('15','20','25'))
ylabel('States')
cbar = colorbar()
cbar.ax.set_ylabel('Distance Ratio')
grid(True)

figure(3)
imshow(A, cmap=plt.cm.gray, interpolation='nearest')
title('All High (10k) Comparison of Distances A:\n Ratio to the Lowest (1.0)')
xticks(arange(4),('10','30','60','90'))
xlabel('Truncation Level')
yticks(arange(3),('15','20','25'))
ylabel('States')
cbar = colorbar()
cbar.ax.set_ylabel('Distance Ratio')
savefig('DisRatio')
grid(True)

figure(4)
imshow(D, cmap=plt.cm.gray, interpolation='nearest')
title('All High (10k) Comparison of Distances D:\n Ratio to the Lowest (1.0)')
xticks(arange(4),('10','30','60','90'))
xlabel('Truncation Level')
yticks(arange(3),('15','20','25'))
ylabel('States')
cbar = colorbar()
cbar.ax.set_ylabel('Distance Ratio')
savefig('DisRatio')
grid(True)

show()
