# HSMM_SimDyad
=============
HSMM based SimDyad is a python project using Bayesian Nonparametric Hidden semi-Markov Models to Disentangle Affect Processes During Marital Interaction
This project is based on Matthew J. Johnson's work <[pyhsmm](https://github.com/mattjj/pyhsmm)>. 

Authors: 

[William A. Griffin](https://webapp4.asu.edu/directory/person/11201),Center for Social Dynamics and Complexity, Arizona State University

Xun Li, Arizona State University

Data
-----------

```
.\matAll\
```

Usage
-----------
The code depends on wxpython, pyhsmm, numpy, scipy.

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

We also have setup 5 experiements for randomization testing -- check the shell script for details
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
