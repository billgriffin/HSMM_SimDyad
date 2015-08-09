# HSMM_SimDyad
=============
HSMM based SimDyad is a python project using Bayesian Nonparametric Hidden semi-Markov Models to Disentangle Affect Processes During Marital Interaction
This project is based on Matthew J. Johnson's work <[pyhsmm](https://github.com/mattjj/pyhsmm)>. 

Note: HSMM_SimDyad is not based on a fork of [pyhsmm](https://github.com/mattjj/pyhsmm). Instead, an old version of [pyhsmm](https://github.com/mattjj/pyhsmm) has been modified to achieve the specific tasks of HSMM SimDyad.

Authors: 

[William A. Griffin](https://webapp4.asu.edu/directory/person/11201),Center for Social Dynamics and Complexity, Arizona State University

[Xun Li](xunli@asu.edu), Arizona State University

Data
-----------
The data used for HSMM_SimDyad is located at:
```
.\matAll\
```
Each file is a sequential data for a specific couple (e.g. c200.txt represents data for c200).

Usage
-----------
Dependencies: wxpython, numpy, scipy.

To run it, just simple call
```
python hsmm.py
```

If you want a GUI, call
```
python hsmmSimDyad.py
```

If you want to run in parallel, call
```
python hsmmHPC.py
```

For the paper "Using Bayesian Nonparametric Hidden semi-Markov Models to Disentangle Affect Processes During Marital Interaction", we have setup 5 experiements for randomization testing -- check the shell script for details
```
python batchScriptRn1.sh
python batchScriptRn2.sh
python batchScriptRn3.sh
python batchScriptRn4.sh
python batchScriptRn5.sh
```

To configure the parameters or hyper-parameters, please ref
```
./conf/gen_params.py
./gcp/couple_gaussianprior.py
```
