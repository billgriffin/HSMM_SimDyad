from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import brewer2mpl

import seaborn as sns
sns.set(style='white')

xx_colors = sns.color_palette()[:1]
xy_colors = sns.color_palette()[2:3]
xx_colors_rgb = ()
xy_colors_rgb = ()
for i in xx_colors:
   xx_colors_rgb += i
for j in xy_colors:
   xy_colors_rgb += j


#set1_color = brewer2mpl.get_map('set1', 'qualitative', 3)  #red and blue
#xy_color = set1_color[1] #  #7570b3
#xx_color = set1_color[0] #  #d95f02

#cmapM=mpl.colors.ListedColormap(list(xx_colors))
#cmapF=mpl.colors.ListedColormap(list(xy_colors))

sub_state_seq_X = (1,2,3,2,3,4)
sub_state_seq_Y = (1,3,3,3,1,4)

plt.figure(figsize=(15, 10))
#plt.plot(sub_state_seq_X, label = 'M', linewidth=2, color= (0.2980392156862745, 0.4470588235294118, 0.6901960784313725) )
#plt.plot(sub_state_seq_Y, label = 'F', linewidth=2, color= (0.7686274509803922, 0.3058823529411765, 0.3215686274509804) )
#plt.plot(sub_state_seq_X, label = 'M', linewidth=2, color= xx_colors_rgb)
#plt.plot(sub_state_seq_Y, label = 'F', linewidth=2, color= xy_colors_rgb)

plt.plot(sub_state_seq_X, label = 'M', linewidth=2.5, color= '#7570b3')
plt.plot(sub_state_seq_Y, label = 'F', linewidth=2.5, color= '#d95f02')


#plt.title('Plausible Sequence of States for individual subject %s in Group: %s'%(index,self.couple_type))


#sns.palplot(xy_colors)

plt.show()