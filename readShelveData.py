import shelve
aff = shelve.open('results/simCoupleAffect.db')

for k,v in sorted(aff.items()):
   print k, v