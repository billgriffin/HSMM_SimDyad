import os, pickle
from os import listdir, remove
from shutil import copyfile
from os.path import isfile, join
import commands
import numpy as np

def GetFiles(mypath):
    onlyfiles = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath,f)) ]
    return onlyfiles

"""
SimCouples data structure
{
  "params1":
  {
    "202":
      {
        "male": [2,3,4,...,2],
        "female": [2,3,4,...,2],
        "combine": [2,3,4,...,2]
      },
    "212":
      {
        "male": [2,3,4,...,2],
        "female": [2,3,4,...,2],
        "combine": [2,3,4,...,2]
      },...
  },
  ...
}
"""

#========================================================
# Get raw couple data
#========================================================
def GetRawCoupleData():
    raw_data = {}
    rawFiles = GetFiles("./matAll")
    for fname in rawFiles:
        if fname.endswith(".txt"):
            file_id = fname.split("/")[-1][1:-4]
            dyad_data = np.loadtxt(fname)
            male = []
            female = []
            States = []
            for i in range(len(dyad_data)):
                male.append(int(dyad_data[i][0]))
                female.append(int(dyad_data[i][1]))
                States.append(male[-1]+female[-1] - 1)
            couple_data = {'male': male, 'female': female, 'States': States}
            raw_data[file_id] = couple_data
    return raw_data

rawCouples = GetRawCoupleData()

#========================================================
# Get simulated couple data
#========================================================
def GetSimCoupleData(path):
    couples = {}
    simFiles = GetFiles(path)
    for fname in simFiles:
        if fname.endswith("csv"):
            param = "_".join(fname.split("/")[-1].split("_")[2:-1])
            if not couples.has_key(param):
                couples[param] = {} 
            f = open(fname)
            line = f.readline()
            while len(line) > 0:
                if line.startswith("Couple:"):
                    coupleIDs = line[line.index('[') + 1: line.index(']')].split(',')

                if len(line) > 0 and line[0].isdigit():
                    idx, timeStep, maleAffect, femaleAffect, State= \
                            line.strip().split(',')
                    coupleID = coupleIDs[int(idx)].strip()
                    if not couples[param].has_key(coupleID):
                        couples[param][coupleID] = {"male":[],"female":[],"States":[]}
                    couples[param][coupleID]["male"].append(int(maleAffect))
                    couples[param][coupleID]["female"].append(int(femaleAffect))
                    couples[param][coupleID]["States"].append(int(State))

                line = f.readline()
            f.close()
    return couples

#========================================================
# Compare all
#========================================================
def compare():
    runs = [1,2,3,4,5]
    clusters = [1,2,3,4,5]
    """
    {
      run1:
        {
          param1:
            {
                202:
                  {
                    "HAM": [127.0, 240.0, 123.0],
                    "DHD": [127.0, 240.0, 123.0],
                    "LCP": [127.0, 240.0, 123.0],
                    "LCS": [127.0, 240.0, 123.0],
                    "OM": [127.0, 240.0, 123.0],
                  },
                  ...
            },
          ...
        },
      run2:
        ...
    }
    
    """
    compare2raw_results = {}
    
    for each_run in runs:
        for cluster in clusters:
            dir_name = "./results%s/Cluster%d/Run%d/Affect" % \
                (each_run, cluster, each_run)
            simCouples = GetSimCoupleData(dir_name)
            for param in simCouples.keys():
                if not param in compare2raw_results.keys():
                    compare2raw_results[param] = {'run%d'%i:{} for i in runs}
                    
                for couple_id in simCouples[param].keys():
                    sim_couple = simCouples[param][couple_id]
                    raw_couple = rawCouples[couple_id]
        
                    def run_compare(t):
                        n = len(raw_couple[t])
                        if n > len(sim_couple[t]): n = len(sim_couple[t])
                        fname = "./compare/run%d_%s_c%s_%s.csv" % (each_run, param, couple_id, t)
                        o = open(fname, "w")
                        header = ",".join(["T%d"%i for i in range(n)])
                        o.write("%s\n" % header)
                        raw_line = ",".join([str(i) for i in raw_couple[t]])
                        o.write("%s\n" % raw_line)
                        sim_line = ",".join([str(sim_couple[t][i]) for i in range(n)])
                        o.write("%s\n" % sim_line)
                        o.close()
                        
                        copyfile(fname, "./tmp.csv")
                        commands.getstatusoutput("R CMD BATCH comp_tramine.R")
                        try:
                            with file("./HAM.csv") as f: ham = float(f.read().strip())
                            with file("./DHD.csv") as f: dhd = float(f.read().strip())
                            with file("./LCP.csv") as f: lcp = float(f.read().strip())
                            with file("./LCS.csv") as f: lcs = float(f.read().strip())
                            #with file("./OM.csv") as f: om= float(f.read().strip())
                            os.remove("./tmp.csv")
                            os.remove("./HAM.csv")
                            os.remove("./DHD.csv")
                            os.remove("./LCP.csv")
                            os.remove("./LCS.csv")
                            #os.remove("./om.csv")
                            return ham, dhd, lcp, lcs
                        except:
                            return 0.0, 0.0, 0.0, 0.0 
                        
                        return 0.0, 0.0, 0.0, 0.0 
    
        
                    run_id =  'run%d'%each_run
                    compare2raw_results[param][run_id][couple_id] = { "HAM":[], "DHD": [], "LCP": [], "LCS": [], "OM": [] }
        
                    ham1, dhd1, lcp1, lcs1 = run_compare("male")
                    ham2, dhd2, lcp2, lcs2 = run_compare("female")
                    ham3, dhd3, lcp3, lcs3 = run_compare("States")
                    compare2raw_results[param][run_id][couple_id] = {
                        "HAM":[ham1, ham2, ham3],
                        "DHD": [dhd1, dhd2, dhd3],
                        "LCP": [lcp1, lcp2, lcp3],
                        "LCS": [lcs1, lcs2, lcs3],
                    }
    pickle.dump( compare2raw_results, open( "save.pickle", "wb" ) )
    
    o = open("save.csv", "w")
    o.write("param,run,couple,DHD.male,DHD.female,DHD.States,HAM.male,HAM.female,HAM.States,LCP.male,LCP.female,LCP.States\n")
    for param in compare2raw_results.keys():
        for run in compare2raw_results[param].keys():
            for couple in compare2raw_results[param][run].keys():
                vals = compare2raw_results[param][run][couple]
                dhd = vals["DHD"]
                ham = vals["HAM"]
                lcp = vals["LCP"]
                o.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % \
                        (param, run, couple, dhd[0], dhd[1], dhd[2],
                         ham[0], ham[1], ham[2], lcp[0], lcp[1], lcp[2]))
    o.close()

#========================================================
# Compare reEsti
#========================================================
#def compare_reEst():
    #compare2raw_results = {}
    
    #dir_name = "./reEstimate"
    #simCouples = GetSimCoupleData(dir_name)
    #for param in simCouples.keys():
        #if not param in compare2raw_results.keys():
            #compare2raw_results[param] = {}
            
        #for couple_id in simCouples[param].keys():
            #sim_couple = simCouples[param][couple_id]
            #raw_couple = rawCouples[couple_id]

            #def run_compare(t):
                #n = len(raw_couple[t])
                #if n > len(sim_couple[t]): n = len(sim_couple[t])
                #fname = "./compare/runX_%s_c%s_%s.csv" % (param, couple_id, t)
                #o = open(fname, "w")
                #header = ",".join(["T%d"%i for i in range(n)])
                #o.write("%s\n" % header)
                #raw_line = ",".join([str(i) for i in raw_couple[t]])
                #o.write("%s\n" % raw_line)
                #sim_line = ",".join([str(sim_couple[t][i]) for i in range(n)])
                #o.write("%s\n" % sim_line)
                #o.close()
                #copyfile(fname, "./tmp.csv")
                #commands.getstatusoutput("R CMD BATCH comp_tramine.R")
                #try:
                    #with file("./HAM.csv") as f: ham = float(f.read().strip())
                    #with file("./DHD.csv") as f: dhd = float(f.read().strip())
                    #with file("./LCP.csv") as f: lcp = float(f.read().strip())
                    #with file("./LCS.csv") as f: lcs = float(f.read().strip())
                    ##with file("./OM.csv") as f: om= float(f.read().strip())
                    #os.remove("./tmp.csv")
                    #os.remove("./HAM.csv")
                    #os.remove("./DHD.csv")
                    #os.remove("./LCP.csv")
                    #os.remove("./LCS.csv")
                    ##os.remove("./om.csv")
                    #return ham, dhd, lcp, lcs
                #except:
                    #return 0.0, 0.0, 0.0, 0.0 

            #compare2raw_results[param][couple_id] = { \
                #"HAM":[], "DHD": [], "LCP": [], "LCS": [], "OM": [] }

            #ham1, dhd1, lcp1, lcs1 = run_compare("male")
            #ham2, dhd2, lcp2, lcs2 = run_compare("female")
            #ham3, dhd3, lcp3, lcs3 = run_compare("States")
            #compare2raw_results[param][couple_id] = {
                #"HAM":[ham1, ham2, ham3],
                #"DHD": [dhd1, dhd2, dhd3],
                #"LCP": [lcp1, lcp2, lcp3],
                #"LCS": [lcs1, lcs2, lcs3],
            #}
    #pickle.dump( compare2raw_results, open( "save.est.pickle", "wb" ) )
    
    #o = open("save.est.csv", "w")
    #o.write("param,run,couple,DHD.male,DHD.female,DHD.States,HAM.male,HAM.female,HAM.States,LCP.male,LCP.female,LCP.States\n")
    #for param in compare2raw_results.keys():
        #for couple in compare2raw_results[param].keys():
            #vals = compare2raw_results[param][couple]
            #dhd = vals["DHD"]
            #ham = vals["HAM"]
            #lcp = vals["LCP"]
            #o.write("%s,runX,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % \
                    #(param, couple, dhd[0], dhd[1], dhd[2],
                     #ham[0], ham[1], ham[2], lcp[0], lcp[1], lcp[2]))
    #o.close()
    
if __name__ == "__main__":
    compare()