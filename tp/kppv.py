import numpy as np
from matplotlib import pyplot as plt

def kppv(X_train, y_train, X_test, k=1):
    myClasses = np.unique(y_train)
    cpt=0
    allNeighbors = {}
    for i in X_test:
        
        thisDists = []
        # Calculate distances between X_test and all X_train 
        for j in X_train:
            thisDists.append(dist_eucl(i, j))
        
        # Find the k closest neighbors
        # Get their index
        allIndexSorted = sorted(range(len(thisDists)), key=lambda k: thisDists[k])
        kIndex = allIndexSorted[0:k]
        # Get their class 
        allNeighbors[cpt] = list(y_train[kIndex])

        # Put results into scores kwj/k
        scores=[]
        for j in np.unique(y_train):
            # Count occurances of class j in the k neighbors
            score = (allNeighbors[cpt] == j).sum()/k
            scores.append(score)
            
        allNeighbors[cpt] = scores
        cpt+=1
    return allNeighbors


def apply_kppv_to_data(data=1, verbrose = True):
    dic={}
    if verbrose:
        print("Applying to data", data, "\n")
    filename = "Archive/data_tp" + str(data)
    X_train, y_train = read_file(filename + '_app.txt')
    X_test, y_test = read_file(filename+'_dec.txt')
    
    if verbrose:
        print("### 1-ppv :")
    allDist = kppv(X_train, y_train, X_test, k=1)
    y_classif = get_classification(allDist, np.unique(y_train))
    dic["1-ppv"] = evaluate_performances(X_train, y_train, 
                                         X_test, y_test, 
                                         y_classif, 
                                         allDist, verbrose=verbrose)
    
    if verbrose:
        dic["1-ppv"]["plot"]
        
    return dic

def apply_CV_kppv_to_data(data=1, k_range = range(1, 5), verbrose = True):
    dic={}
    if verbrose:
        print("Applying to data", data, "\n")
    filename = "Archive/data_tp" + str(data)
    X_train, y_train = read_file(filename + '_app.txt')
    X_test, y_test = read_file(filename+'_dec.txt')
    
    dic = CrossValidate(X_train, y_train, k_range = k_range, verbrose=verbrose)
        
    return dic