import numpy as np
import itertools

def isplit(iterable,spliters):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in spliters) if not k]

def printplus(obj):
    """
    Pretty-prints the object passed in.

    """
    # Dict
    if isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            print u'{0}: {1}'.format(k, v)

    # List or tuple            
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for x in obj:
            print x

    # Other
    else:
        print obj


