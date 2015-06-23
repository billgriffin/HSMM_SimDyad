import os
import subprocess

clusters = [1,2,3,4,5]
runs = [1,2,3,4,5]
couple_dict = {1:[203,212,217,225],
           2:[204,205,208,209,210,211,214,215,219,224],
           3:[206,207,218,220,222,228,230],
           4:[200,202, 213],
           5:[201,221,223,226,227,229]}
dir_name = 'Cluster%d/Run%d'
current_dir = os.path.dirname(os.path.realpath(__file__))
command_dir = os.path.split(current_dir)[0]

for i in clusters:
    for j in runs:
        os.chdir(current_dir)
        dir_path = dir_name % (i, j)
        items = os.listdir(dir_path) 
        os.chdir(os.path.join(current_dir, 'Cluster%d'%i, 'Run%d'%j))
        for f in items:
            if f.endswith(".txt"):
                #conf/parameter36.conf results4/Summary/All_Cluster2_36_10000_4.0_8.0_20_90.txt 1 
                param_idx = f.split("_")[2]
                gen_path = os.path.join(command_dir, "GenHSMMGraphs.py")
                param_path = os.path.join(command_dir, "conf", "parameter%s.conf" % param_idx)
                result_path = os.path.join(command_dir, "results%d" % j, "Summary", f)
                command = 'python %s %s %s %d'
                couples = couple_dict[i]
                for k in range(len(couples)):
                    cmd = command % (gen_path, param_path, result_path, k+1)
                    subprocess.call(cmd)
                    
                