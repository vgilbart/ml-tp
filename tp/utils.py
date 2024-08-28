import numpy as np
from matplotlib import pyplot as plt

def read_file(file):
    val = np.loadtxt(file, delimiter=' ')
    return val[:,1:3], val[:,0] #X and y

def get_classification(allDist, myClasses, vote='max'):
    c = []
    for pt in allDist:
        maxVal = max(allDist[pt])
        classifPosition = np.where(allDist[pt]==maxVal)[0][0]
        thisClass = myClasses[classifPosition]
        if vote == 'unanimity':
            if maxVal != 1:
                thisClass = None
        elif vote == 'majority':
            if maxVal <= 0.5:
                thisClass = None
        c.append(thisClass)
    return c