from matplotlib import pyplot as plt
import numpy as np

files = [
    'All_Cluster1_6_10000_4.0_4.0_20_90_sim_generic.txt',
    'All_Cluster2_30_10000_4.0_4.0_20_90_sim_generic.txt',
    'All_Cluster3_54_10000_4.0_4.0_20_90_sim_generic.txt',
    'All_Cluster4_78_10000_4.0_4.0_20_90_sim_generic.txt',
    'All_Cluster5_102_10000_4.0_4.0_20_90_sim_generic.txt',
]

data = []
data_detail = []

for f in files:
    x_seq = []
    y_seq = []
    ff = open(f)
    line = ff.readline()
    while len(line) > 0:
        line = line.strip()
        x = line[:line.index(",")]
        x_seq.append(float(x))
        y_seq = line[line.index(",")+1:]
        line = ff.readline()
    ff.close()
   
    y_seq = eval(y_seq)
    x_seq = x_seq[:700] 
    y_seq = y_seq[:700] 
    seq = [x_seq[i] + y_seq[i] - 1 for i in range(len(x_seq))]
    data.append(seq)
    data_detail.append([
        [x_seq[i] for i in range(len(x_seq))],
        [y_seq[i] for i in range(len(x_seq))],
    ])

plt.figure(figsize=(15, 10))

from matplotlib import cm
import matplotlib.patches as mpatches

colors = []
values = []
cmap = cm.get_cmap('jet')
for i in range(5):
    avg = sum(data[i])/len(data[i])
    values.append(avg)
    colors.append(cmap(avg/10.0))
    
# seaborn colorpalette default 5 of 6
colors = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.6980392156862745),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]

"""
ax0 = plt.subplot(5,1,1)

def label(xy, text):
    y = xy[1] - 0.15 # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=8)
    
grid = np.mgrid[0.1:0.9:20j, 0.2:0.2:1j].reshape(2, -1).T 
w = 0.03
h = 0.1
offset = [0.015, 0.05]

for i in range(2,7):
    rect = mpatches.Rectangle(grid[i]-offset, w, h, ec="none",color=colors[i-2])
    label(grid[i], "%.2f" % (values[i-2]))
    ax0.add_patch(rect)

plt.xlim ((0.05, 0.95))
plt.ylim ((0, 0.7))

plt.text(0.12, 0.05, "Average State Score:", ha="center", family='sans-serif', size=8)
plt.axis('off')    
"""
ax1 = plt.subplot(5,1,1)
plt.plot(data[0], label = 'cluster 1', linewidth=2, color = colors[0] )
plt.title('Cluster 1')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax1.set_ylabel("States")
#ax1.get_xaxis().set_visible(False)

ax2 = plt.subplot(5,1,2)
plt.plot(data[1], label = 'cluster 2', linewidth=2, color = colors[1] )
plt.title('Cluster 2')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax2.set_ylabel("States")
#ax2.get_xaxis().set_visible(False)

ax3 = plt.subplot(5,1,3)
plt.plot(data[2], label = 'cluster 3', linewidth=2, color = colors[2] )
plt.title('Cluster 3')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax3.set_ylabel("States")
#ax3.get_xaxis().set_visible(False)

ax4 = plt.subplot(5,1,4)
plt.plot(data[3], label = 'cluster 4', linewidth=2, color = colors[3] )
plt.title('Cluster 4')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax4.set_ylabel("States")
#ax4.get_xaxis().set_visible(False)

ax5 = plt.subplot(5,1,5)
plt.plot(data[4], label = 'cluster 5', linewidth=2, color = colors[4] )
plt.title('Cluster 5')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax5.set_ylabel("States")
ax5.set_xlabel("Time")

#plt.legend()
plt.show()


plt.figure(figsize=(15, 10))

ax1 = plt.subplot(5,1,1)
plt.plot(data_detail[0][0], label='Male', color = 'b' )
plt.plot(data_detail[0][1], label='Female', color = 'r' )
plt.title('c1')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax1.get_xaxis().set_visible(False)
plt.legend()

ax2 = plt.subplot(5,1,2)
plt.plot(data_detail[1][0], label='Male', color = 'b' )
plt.plot(data_detail[1][1], label='Female', color = 'r' )
plt.title('c2')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax2.get_xaxis().set_visible(False)

ax3 = plt.subplot(5,1,3)
plt.plot(data_detail[2][0], label='Male', color = 'b' )
plt.plot(data_detail[2][1], label='Female', color = 'r' )
plt.title('c3')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax3.get_xaxis().set_visible(False)

ax4 = plt.subplot(5,1,4)
plt.plot(data_detail[3][0], label='Male', color = 'b' )
plt.plot(data_detail[3][1], label='Female', color = 'r' )
plt.title('c4')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))
ax4.get_xaxis().set_visible(False)

ax5 = plt.subplot(5,1,5)
plt.plot(data_detail[4][0], label='Male', color = 'b' )
plt.plot(data_detail[4][1], label='Female', color = 'r' )
plt.title('c5')
plt.xlim((0, len(data[0])))
plt.ylim((0, 10))

#plt.legend()
plt.show()