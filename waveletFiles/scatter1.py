""" Make sure that imported file contains no ,, gaps"""

import os
import numpy as np
from pylab import *
import copy

np.set_printoptions(precision = 3)

satlevel,cluster,Id,states10,states20,states30,states40 = np.loadtxt(
	'wavedata.txt', unpack = True)

N = len(satlevel) # get number of occurences
size = []; colors = []; markers = []

## Dyad grouping
MAT1 = []; MAT2 = []; MAT3 =[]; cluster1 = []; cluster2 = []
states_10 = []; states_20 = []; states_30 = []; states_40 = []

MAT1Cluster1_10 = [];MAT1Cluster2_10 = []
MAT1Cluster1_20 = [];MAT1Cluster2_20 = []
MAT1Cluster1_30 = [];MAT1Cluster2_30 = []
MAT1Cluster1_40 = [];MAT1Cluster2_40 = []

MAT2Cluster1_10 = [];MAT2Cluster2_10 = []
MAT2Cluster1_20 = [];MAT2Cluster2_20 = []
MAT2Cluster1_30 = [];MAT2Cluster2_30 = []
MAT2Cluster1_40 = [];MAT2Cluster2_40 = []

MAT3Cluster1_10 = [];MAT3Cluster2_10 = []
MAT3Cluster1_20 = [];MAT3Cluster2_20 = []
MAT3Cluster1_30 = [];MAT3Cluster2_30 = []
MAT3Cluster1_40 = [];MAT3Cluster2_40 = []



#Pos1 = []; VecLen1 = []; grpDyad1 = []; grpDyad2 = []; grpDyad3 = []
#grp1Den1 = []; grp2Den1 = []; grp3Den1 = []; grp1Pos1 = []; grp2Pos1 = []; grp3Pos1 = []


# generate size of symbol by group
grpSize1 = N * [15.**2]
grpSize0 = N * [10.**2]

# generate subgroups for analysis and graphing

for each in range(len(satlevel)):
	if satlevel[each] == 1.00:
		MAT1.append(satlevel[each])
		if cluster[each] == 1.00:
			MAT1Cluster1_10.append(states10[each])
			MAT1Cluster1_20.append(states20[each])
			MAT1Cluster1_30.append(states30[each])
			MAT1Cluster1_40.append(states40[each])	
			
		if cluster[each] == 2.00:
			MAT1Cluster2_10.append(states10[each])
			MAT1Cluster2_20.append(states20[each])
			MAT1Cluster2_30.append(states30[each])
			MAT1Cluster2_40.append(states40[each])
	elif satlevel[each] == 2.00:
		MAT2.append(satlevel[each])
		if cluster[each] == 1.00:
			MAT2Cluster1_10.append(states10[each])
			MAT2Cluster1_20.append(states20[each])
			MAT2Cluster1_30.append(states30[each])
			MAT2Cluster1_40.append(states40[each])	
			
		if cluster[each] == 2.00:
			MAT2Cluster2_10.append(states10[each])
			MAT2Cluster2_20.append(states20[each])
			MAT2Cluster2_30.append(states30[each])
			MAT2Cluster2_40.append(states40[each])		
	else:
		if cluster[each] == 1.00:
			MAT3Cluster1_10.append(states10[each])
			MAT3Cluster1_20.append(states20[each])
			MAT3Cluster1_30.append(states30[each])
			MAT3Cluster1_40.append(states40[each])	
			
		if cluster[each] == 2.00:
			MAT3Cluster2_10.append(states10[each])
			MAT3Cluster2_20.append(states20[each])
			MAT3Cluster2_30.append(states30[each])
			MAT3Cluster2_40.append(states40[each])	

grouplabels = ['MAT1Cluster1_10','MAT1Cluster2_10',
               'MAT1Cluster1_20','MAT1Cluster2_20',
               'MAT1Cluster1_30','MAT1Cluster2_30',
               'MAT1Cluster1_40','MAT1Cluster2_40',
               'MAT2Cluster1_10','MAT2Cluster2_10',
               'MAT2Cluster1_20','MAT2Cluster2_20',
               'MAT2Cluster1_30','MAT2Cluster2_30',
               'MAT2Cluster1_40','MAT2Cluster2_40',
               'MAT3Cluster1_10','MAT3Cluster2_10',
               'MAT3Cluster1_20','MAT3Cluster2_20',
               'MAT3Cluster1_30','MAT3Cluster2_30',
               'MAT3Cluster1_40','MAT3Cluster2_40']

groups = [MAT1Cluster1_10,MAT1Cluster2_10,
          MAT1Cluster1_20,MAT1Cluster2_20,
          MAT1Cluster1_30,MAT1Cluster2_30,
          MAT1Cluster1_40,MAT1Cluster2_40,
          MAT2Cluster1_10,MAT2Cluster2_10,
          MAT2Cluster1_20,MAT2Cluster2_20,
          MAT2Cluster1_30,MAT2Cluster2_30,
          MAT2Cluster1_40,MAT2Cluster2_40,
          MAT3Cluster1_10,MAT3Cluster2_10,
          MAT3Cluster1_20,MAT3Cluster2_20,
          MAT3Cluster1_30,MAT3Cluster2_30,
          MAT3Cluster1_40,MAT3Cluster2_40]
for i in range(len(groups)):
	print 'Group: %s Freq: %i Vector Length mean: %1.3f sd: %1.3f median: %1.3f'%(
		grouplabels[i], len(groups[i]), np.mean(groups[i]), np.std(groups[i]), np.median(groups[i]))


#sortVecLen1 = copy.deepcopy(VecLen1)
#sortVecLen1.sort()
#xVecLen1 = np.mean(VecLen1)
#yVecLen1 = np.std(VecLen1)

#sortVecLen0 = copy.deepcopy(VecLen0)
#sortVecLen0.sort()
#xVecLen0 = np.mean(VecLen0)
#yVecLen0 = np.std(VecLen0)


#grpMeans1 = (np.mean(grpDyad1), np.mean(grpDyad2), np.mean(grpDyad3))
#grpStd1 =   (np.std(grpDyad1), np.std(grpDyad2), np.std(grpDyad3))

#grpMeans0 = (np.mean(grpNoDyad1),np.mean(grpNoDyad2),np.mean(grpNoDyad3))
#grpStd0 =   (np.std(grpNoDyad1),np.std(grpNoDyad2),np.std(grpNoDyad3))


"""cnames = {
    'aliceblue'            : '#F0F8FF',
    'antiquewhite'         : '#FAEBD7',
    'aqua'                 : '#00FFFF',
    'aquamarine'           : '#7FFFD4',
    'azure'                : '#F0FFFF',
    'beige'                : '#F5F5DC',
    'bisque'               : '#FFE4C4',
    'black'                : '#000000',
    'blanchedalmond'       : '#FFEBCD',
    'blue'                 : '#0000FF',
    'blueviolet'           : '#8A2BE2',
    'brown'                : '#A52A2A',
    'burlywood'            : '#DEB887',
    'cadetblue'            : '#5F9EA0',
    'chartreuse'           : '#7FFF00',
    'chocolate'            : '#D2691E',
    'coral'                : '#FF7F50',
    'cornflowerblue'       : '#6495ED',
    'cornsilk'             : '#FFF8DC',
    'crimson'              : '#DC143C',
    'cyan'                 : '#00FFFF',
    'darkblue'             : '#00008B',
    'darkcyan'             : '#008B8B',
    'darkgoldenrod'        : '#B8860B',
    'darkgray'             : '#A9A9A9',
    'darkgreen'            : '#006400',
    'darkkhaki'            : '#BDB76B',
    'darkmagenta'          : '#8B008B',
    'darkolivegreen'       : '#556B2F',
    'darkorange'           : '#FF8C00',
    'darkorchid'           : '#9932CC',
    'darkred'              : '#8B0000',
    'darksalmon'           : '#E9967A',
    'darkseagreen'         : '#8FBC8F',
    'darkslateblue'        : '#483D8B',
    'darkslategray'        : '#2F4F4F',
    'darkturquoise'        : '#00CED1',
    'darkviolet'           : '#9400D3',
    'deeppink'             : '#FF1493',
    'deepskyblue'          : '#00BFFF',
    'dimgray'              : '#696969',
    'dodgerblue'           : '#1E90FF',
    'firebrick'            : '#B22222',
    'floralwhite'          : '#FFFAF0',
    'forestgreen'          : '#228B22',
    'fuchsia'              : '#FF00FF',
    'gainsboro'            : '#DCDCDC',
    'ghostwhite'           : '#F8F8FF',
    'gold'                 : '#FFD700',
    'goldenrod'            : '#DAA520',
    'gray'                 : '#808080',
    'green'                : '#008000',
    'greenyellow'          : '#ADFF2F',
    'honeydew'             : '#F0FFF0',
    'hotpink'              : '#FF69B4',
    'indianred'            : '#CD5C5C',
    'indigo'               : '#4B0082',
    'ivory'                : '#FFFFF0',
    'khaki'                : '#F0E68C',
    'lavender'             : '#E6E6FA',
    'lavenderblush'        : '#FFF0F5',
    'lawngreen'            : '#7CFC00',
    'lemonchiffon'         : '#FFFACD',
    'lightblue'            : '#ADD8E6',
    'lightcoral'           : '#F08080',
    'lightcyan'            : '#E0FFFF',
    'lightgoldenrodyellow' : '#FAFAD2',
    'lightgreen'           : '#90EE90',
    'lightgrey'            : '#D3D3D3',
    'lightpink'            : '#FFB6C1',
    'lightsalmon'          : '#FFA07A',
    'lightseagreen'        : '#20B2AA',
    'lightskyblue'         : '#87CEFA',
    'lightslategray'       : '#778899',
    'lightsteelblue'       : '#B0C4DE',
    'lightyellow'          : '#FFFFE0',
    'lime'                 : '#00FF00',
    'limegreen'            : '#32CD32',
    'linen'                : '#FAF0E6',
    'magenta'              : '#FF00FF',
    'maroon'               : '#800000',
    'mediumaquamarine'     : '#66CDAA',
    'mediumblue'           : '#0000CD',
    'mediumorchid'         : '#BA55D3',
    'mediumpurple'         : '#9370DB',
    'mediumseagreen'       : '#3CB371',
    'mediumslateblue'      : '#7B68EE',
    'mediumspringgreen'    : '#00FA9A',
    'mediumturquoise'      : '#48D1CC',
    'mediumvioletred'      : '#C71585',
    'midnightblue'         : '#191970',
    'mintcream'            : '#F5FFFA',
    'mistyrose'            : '#FFE4E1',
    'moccasin'             : '#FFE4B5',
    'navajowhite'          : '#FFDEAD',
    'navy'                 : '#000080',
    'oldlace'              : '#FDF5E6',
    'olive'                : '#808000',
    'olivedrab'            : '#6B8E23',
    'orange'               : '#FFA500',
    'orangered'            : '#FF4500',
    'orchid'               : '#DA70D6',
    'palegoldenrod'        : '#EEE8AA',
    'palegreen'            : '#98FB98',
    'palevioletred'        : '#AFEEEE',
    'papayawhip'           : '#FFEFD5',
    'peachpuff'            : '#FFDAB9',
    'peru'                 : '#CD853F',
    'pink'                 : '#FFC0CB',
    'plum'                 : '#DDA0DD',
    'powderblue'           : '#B0E0E6',
    'purple'               : '#800080',
    'red'                  : '#FF0000',
    'rosybrown'            : '#BC8F8F',
    'royalblue'            : '#4169E1',
    'saddlebrown'          : '#8B4513',
    'salmon'               : '#FA8072',
    'sandybrown'           : '#FAA460',
    'seagreen'             : '#2E8B57',
    'seashell'             : '#FFF5EE',
    'sienna'               : '#A0522D',
    'silver'               : '#C0C0C0',
    'skyblue'              : '#87CEEB',
    'slateblue'            : '#6A5ACD',
    'slategray'            : '#708090',
    'snow'                 : '#FFFAFA',
    'springgreen'          : '#00FF7F',
    'steelblue'            : '#4682B4',
    'tan'                  : '#D2B48C',
    'teal'                 : '#008080',
    'thistle'              : '#D8BFD8',
    'tomato'               : '#FF6347',
    'turquoise'            : '#40E0D0',
    'violet'               : '#EE82EE',
    'wheat'                : '#F5DEB3',
    'white'                : '#FFFFFF',
    'whitesmoke'           : '#F5F5F5',
    'yellow'               : '#FFFF00',
    'yellowgreen'          : '#9ACD32',
    }"""

figure(1)
## States: 10
scatter(MAT1Cluster1_10, MAT1Cluster1_20, s = 15.0**2, c = 'darkblue',  
		marker = '^', alpha = 0.75, label = 'States: 10 v 20')
scatter(MAT1Cluster1_10, MAT1Cluster1_30, s = 15.0**2, c = 'steelblue', 
		marker = 'o', alpha = 0.75, label = 'States: 10 v 30')
scatter(MAT1Cluster1_10, MAT1Cluster1_40, s = 15.0**2, c = 'silver',    
		marker = 'd', alpha = 0.75, label = 'States: 10 v 40')

xlabel(r'10 States', fontsize=20)
ylabel(r'States', fontsize=20)
#title('Density By Proportion Positive', fontsize=20)
grid(True)
#legend()
legend(loc=1, shadow = True) #handlelen = .1, )

figure(2)
## States by level
scatter(states10, satlevel, s = 10.0**2, c = 'wheat',      
		marker ='^', alpha=0.75, label = 'States: 10')
scatter(states20, satlevel, s = 10.0**2, c = 'sandybrown', 
		marker ='o', alpha=0.75, label = 'States: 20')

xlabel(r'States', fontsize=20)
ylabel(r'MAT', fontsize=20)
#title('Density By Proportion Positive', fontsize=20)
grid(True)
#legend()
legend(loc=1, shadow = True) #handlelen = .1, )


#figure(3)
#scatter(MAT1Cluster1_30, MAT1Cluster1_40, s = 10.0**2, c = 'salmon',     
		#marker ='d', alpha=0.75, label = 'States: 30 v 40')

#xlabel(r'30 States', fontsize=20)
#ylabel(r'States', fontsize=20)
##title('Density By Proportion Positive', fontsize=20)
#grid(True)
##legend()
#legend(loc=1, shadow = True) #handlelen = .1, )

figure(3)
scatter(MAT1Cluster2_10, MAT1Cluster2_20, s = 10.0**2, c = 'salmon',     
		marker ='d', alpha=0.75, label = 'States: 10 v 20')

xlabel(r'MAT 1, C2', fontsize=20)
ylabel(r'MAT 1, C2 20', fontsize=20)
#title('Density By Proportion Positive', fontsize=20)
grid(True)
#legend()
legend(loc=1, shadow = True) #handlelen = .1, )

figure(4)
scatter(states10, satlevel, s = 15.0**2, c = 'darkblue',  
		marker = '^', alpha = 0.75, label = 'States: 10')
scatter(states20, satlevel, s = 15.0**2, c = 'wheat', 
		marker = 'o', alpha = 0.75, label = 'States: 20')
scatter(states30, satlevel, s = 15.0**2, c = 'silver',    
		marker = 'd', alpha = 0.75, label = 'States: 30')
scatter(states40, satlevel, s = 15.0**2, c = 'salmon',    
		marker = 'd', alpha = 0.75, label = 'States: 40')

xlabel(r'Distance', fontsize=20)
ylabel(r'MAT', fontsize=20)
#title('Density By Proportion Positive', fontsize=20)
grid(True)
#legend()
legend(loc=1, shadow = True) #handlelen = .1, )





###VecLen by grp and dyad level
#figure(2)
##print grpDyad1
##print sort(grpDyad1)
##print
##print grpDyad2
##print sort(grpDyad2)
##print grpDyad3
##print sort(grpDyad3)

#tgrp1 = np.array(range(len( grpDyad1 )))
#tgrp2 = np.array(range(len( grpDyad2 )))
#tgrp3 = np.array(range(len( grpDyad3 )))

#tgrpNo1 = np.array(range(len( grpNoDyad1 )))
#tgrpNo2 = np.array(range(len( grpNoDyad2 )))
#tgrpNo3 = np.array(range(len( grpNoDyad3 )))

#xlabel(r'Occurences', fontsize=20)
#ylabel(r'Vector Length', fontsize=20)
#title('Vector Length by Group and Dyad Status (sorted)', fontsize=20)
#plot( tgrp1, sort(grpDyad1), c = 'darkblue',  marker ='o', markersize=8, label= 'Dyad1')
#plot( tgrp2, sort(grpDyad2), c = 'steelblue', marker ='^', markersize=8, label= 'Dyad2')
#plot( tgrp3, sort(grpDyad3), c = 'silver',    marker ='d', markersize=8, label= 'Dyad3')
#plot( tgrpNo1, sort(grpNoDyad1), c = 'wheat', marker ='o', markersize=8, label= 'NoDyad1')
#plot( tgrpNo2, sort(grpNoDyad2), c = 'sandybrown', marker ='^',markersize=8, label= 'NoDyad2')
#plot( tgrpNo3, sort(grpNoDyad3), c = 'salmon', marker ='d',markersize=8, label= 'NoDyad3')
#legend()

### sorted VecLen score by group
#figure(3)
#t1 = np.array(range(len(VecLen1)))
#t0 = np.array(range(len(VecLen0)))
#xlabel(r'Occurences', fontsize=20)
#ylabel(r'Vector Length', fontsize=20)
#title('Vector Length (sorted)', fontsize=20)
#plot(t1, sortVecLen1, 'bo', label= 'Dyad')
#plot(t0, sortVecLen0, 'ro', label= 'NonDyad')
#legend()


figure(5)
##states 10
MAT1_all_10 = copy.deepcopy(MAT1Cluster1_10)
MAT1_all_10.extend(MAT1Cluster2_10)
MAT1mean10 = np.mean(MAT1_all_10)
MAT1std10 = np.std(MAT1_all_10)

MAT2_all_10 = copy.deepcopy(MAT2Cluster1_10)
MAT2_all_10.extend(MAT2Cluster2_10)
MAT2mean10 = np.mean(MAT2_all_10)
MAT2std10 = np.std(MAT2_all_10)

MAT3_all_10 = copy.deepcopy(MAT3Cluster1_10)
MAT3_all_10.extend(MAT3Cluster2_10)
MAT3mean10 = np.mean(MAT3_all_10)
MAT3std10 = np.std(MAT3_all_10)


#states20
MAT1_all_20 = copy.deepcopy(MAT1Cluster1_20)
MAT1_all_20.extend(MAT1Cluster2_20)
MAT1mean20 = np.mean(MAT1_all_20)
MAT1std20 = np.std(MAT1_all_20)

MAT2_all_20 = copy.deepcopy(MAT2Cluster1_20)
MAT2_all_20.extend(MAT2Cluster2_20)
MAT2mean20 = np.mean(MAT2_all_20)
MAT2std20 = np.std(MAT2_all_20)

MAT3_all_20 = copy.deepcopy(MAT3Cluster1_20)
MAT3_all_20.extend(MAT3Cluster2_20)
MAT3mean20 = np.mean(MAT3_all_20)
MAT3std20 = np.std(MAT3_all_20)

#states30
MAT1_all_30 = copy.deepcopy(MAT1Cluster1_30)
MAT1_all_30.extend(MAT1Cluster2_30)
MAT1mean30 = np.mean(MAT1_all_30)
MAT1std30 = np.std(MAT1_all_30)

MAT2_all_30 = copy.deepcopy(MAT2Cluster1_30)
MAT2_all_30.extend(MAT2Cluster2_30)
MAT2mean30 = np.mean(MAT2_all_30)
MAT2std30 = np.std(MAT2_all_30)

MAT3_all_30 = copy.deepcopy(MAT3Cluster1_30)
MAT3_all_30.extend(MAT3Cluster2_30)
MAT3mean30 = np.mean(MAT3_all_30)
MAT3std30 = np.std(MAT3_all_30)

#states 40
MAT1_all_40 = copy.deepcopy(MAT1Cluster1_40)
MAT1_all_40.extend(MAT1Cluster2_40)
MAT1mean40 = np.mean(MAT1_all_40)
MAT1std40 = np.std(MAT1_all_40)

MAT2_all_40 = copy.deepcopy(MAT2Cluster1_40)
MAT2_all_40.extend(MAT2Cluster2_40)
MAT2mean40 = np.mean(MAT2_all_40)
MAT2std40 = np.std(MAT2_all_40)

MAT3_all_40 = copy.deepcopy(MAT3Cluster1_40)
MAT3_all_40.extend(MAT3Cluster2_40)
MAT3mean40 = np.mean(MAT3_all_40)
MAT3std40 = np.std(MAT3_all_40)



MATmeans10 = (MAT1mean10,MAT2mean10,MAT3mean10)
MATstd10 = (MAT1std10,MAT2std10,MAT2std10)

MATmeans20 = (MAT1mean20,MAT2mean20,MAT3mean20)
MATstd20 = (MAT1std20,MAT2std20,MAT2std20)

MATmeans30 = (MAT1mean30,MAT2mean30,MAT3mean30)
MATstd30 = (MAT1std30,MAT2std30,MAT2std30)

MATmeans40 = (MAT1mean40,MAT2mean40,MAT3mean40)
MATstd40 = (MAT1std40,MAT2std40,MAT2std40)


MAT1means = (MAT1mean10,MAT1mean20,MAT1mean30,MAT1mean40)
MAT1std = (MAT1std10,MAT1std20,MAT1std30,MAT1std40)
#######################################################
MAT1means = (MAT1mean10,MAT1mean20,MAT1mean30,MAT1mean40)
MAT1std =(MAT1std10,MAT1std20,MAT1std30,MAT1std40)

MAT2means = (MAT2mean10,MAT2mean20,MAT2mean30,MAT2mean40)
MAT2std =(MAT2std10,MAT2std20,MAT2std30,MAT2std40)

MAT3means = (MAT3mean10,MAT3mean20,MAT3mean30,MAT3mean40)
MAT3std =(MAT3std10,MAT3std20,MAT3std30,MAT3std40)
#######################################################

N = 3
ind = np.arange(N) 
width = 0.15
#bar means a column in each MAT category
rects1 = bar(ind, MATmeans10, width, color='royalblue', yerr = MATstd10)  
rects2 = bar(ind+width, MATmeans20, width, color='salmon', yerr = MATstd20)
rects3 = bar(ind+(width*2), MATmeans30, width, color='wheat', yerr = MATstd30)
rects4 = bar(ind+(width*3), MATmeans40, width, color='steelblue', yerr = MATstd40)

## add some
ylim(ymin=25)
ylabel('Average Distance')
title('Average Distance (with standard error) by MAT Level and Number of States')
xticks( ind+(width*2), ('High', 'Middle', 'Low') )
legend( (rects1[0], rects2[0],rects3[0], rects4[0]), ('States: 10', 'States: 20','States: 30','States: 40') )

def autolabel(rects):
	# attach some text labels    
	for rect in rects:        
		height = rect.get_height()
		#print errs[rect]
		#err = errs[rect]
		text(rect.get_x()+rect.get_width()/2., 1.005*height, '%.3f'%float(height),
			 ha='center', va='bottom')
		#text(rect.get_x()+rect.get_width()/2., 1.005*height, '%.3f'%float(err),
			#ha='center', va='top')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)


show()