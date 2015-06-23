import scipy
import scipy.stats

files = [
    'matAllstatesDur.csv',
    'matHighGp1statesDur.csv',
    'matHighGp2statesDur.csv',
    'matHighstatesDur.csv',
    'matMedGp1statesDur.csv',
    'matMedGp2statesDur.csv',
    'matMedstatesDur.csv',
    'matLowGp1statesDur.csv',
    'matLowGp2statesDur.csv',
    'matLowstatesDur.csv']
   
o = open('gamma.csv','w')

for fname in files:
    f = open(fname)
    lines = f.readlines()
    f.close()
    
    data = [float(l.strip().split(',')[1]) for l in lines]
    n = len(data)
    for i in range( 9*9 - n):
        data.append(0)
        
    alpha, loc,scale = scipy.stats.gamma.fit(data)
    o.write('%s,%s,%s\n' % (fname, alpha,scale))
o.close()