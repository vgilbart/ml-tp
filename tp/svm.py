import numpy as np
from matplotlib import pyplot as plt

def SVM(X_train, y_train, X_test, error = 0, maxIter = 50, one_vs_one = True):
    
    # Train part
    
    # For each couple of classes, 
    # find an hyperplan that seperates the 2 classes
    # (i.e. is for 5 classes, get 5*4/2 hyperplans)
    myClasses = np.unique(y_train)
    mySize = np.shape(X_train) 
    # mySize[0] is nb of points; mySize[1] is dim of points
    
    # Dictionnary of my hyperplans
    hyperplans = {}

    # Find hyperplan between 2 classes
    nbClasses = np.shape(myClasses)[0]
    for i in range(0, nbClasses):
        if one_vs_one :
            # Get all 2-combinations of classes
            for j in range(0, nbClasses):
                if i < j : 
                    # Refine the data to the one of the 2 classes of interest
                    thisPoints = np.append(np.where(y_train==myClasses[i]) , np.where(y_train==myClasses[j]))
                    thisX_train = X_train[thisPoints,]
                    thisy_train = y_train[thisPoints,]
                    # Calculate hyperplans between the 2 classes of interest
                    hyperplans[str(myClasses[i])+'_'+str(myClasses[j])] = Rosenblatt_perceptron(thisX_train, thisy_train, error = error, maxIter = maxIter)
        else :
            # Modify y so that we only get 2 classes 
            # First the one that we need to separate (named i) 
            # and then the others (renamed -1, or 0 if i is already -1)
            thisy_train = np.copy(y_train)
            otherPoints = np.where(y_train!=myClasses[i])[0]
            thisPoints = np.where(y_train==myClasses[i])[0]
            if myClasses[i] != -1 : 
                np.put(thisy_train, otherPoints, -1)
                thisX_train = np.concatenate([X_train[thisPoints,], X_train[otherPoints,]])
                thisy_train = np.concatenate([thisy_train[thisPoints,], thisy_train[otherPoints,]] )
            else : 
                np.put(thisy_train, otherPoints, 0)
                thisX_train = np.concatenate([X_train[thisPoints,], X_train[otherPoints,]])
                thisy_train = np.concatenate([thisy_train[thisPoints,], thisy_train[otherPoints,]] )
            # Calculate hyperplans between the 2 classes of interest
            hyperplans[str(myClasses[i])+'_vs_all'] = Rosenblatt_perceptron(thisX_train, thisy_train, error = error, maxIter = maxIter)
    
    # Test part 
    
    myTestSize = np.shape(X_test)

    # myTestSize[0] is nb of points to classify;
    # myTestSize[1] is dim of points to classify
    myRes = {}
    # For all points in X_test
    for i in range(0, myTestSize[0]):
        bestA = 0
        thisA = None
        x = X_test[i] # a point in X_test
        l = [] # contains the class of x according to all hyperplans
        for a in hyperplans:
            if hyperplans[a] @ np.append(x, 1) > 0: # point is in class 1
                if one_vs_one:
                    thisClass = a.split(sep='_')[0]
                    l.append(thisClass)
                elif hyperplans[a] @ np.append(x, 1) > bestA: #& not one_vs_one
                    bestA = hyperplans[a] @ np.append(x, 1)
                    thisA = a.split(sep='_')[0]

                else : #not one_vs_one & hyperplans[a] @ np.append(x, 1) <= bestA
                    pass
            elif one_vs_one : # & point is not in class 1
                thisClass = a.split(sep='_')[-1]
                l.append(thisClass)
            else : # not one_vs_one & point is not in class 1
                pass
        if not one_vs_one and thisA != None: 
            thisClass = thisA.split(sep='_')[0]
            l.append(thisClass)           
        # Put results into scores: count(class in l)/len(l)
        scores=[]
        #print(l)
        for j in myClasses:
            # Count occurances of class j in the list
            if len(l) != 0 :
                score = l.count(str(j)) / len(l)
            else: 
                score = 0
            scores.append(score)
        myRes[i] = np.array(scores)
                
    return myRes


def apply_SVM_to_data(data=1, error = 0, maxIter = 50, one_vs_one = True, verbrose = True):
    dic={}
    if verbrose:
        print("Applying to data", data, "\n")
    filename = "Archive/data_tp" + str(data)
    X_train, y_train = read_file(filename + '_app.txt')
    X_test, y_test = read_file(filename+'_dec.txt')
       
    # Removing the last class to be able to do one vs all
    if one_vs_one == False :
        to_remove={'1':4, '2':2, '3':0}
        myClasses = np.unique(y_train)
        thisPoints = np.where(y_train!=myClasses[to_remove[str(data)]])[0]
        X_train = X_train[thisPoints,]
        y_train = y_train[thisPoints,]
        thisPoints = np.where(y_test!=myClasses[to_remove[str(data)]])[0]
        X_test = X_test[thisPoints,]
        y_test = y_test[thisPoints,]


    if verbrose:
        if one_vs_one:
            print("### Linear SVM one_vs_one :")
        else :
            print("### Linear SVM one_vs_all :")
    allDist = SVM(X_train, y_train, X_test, error=error, maxIter=maxIter, one_vs_one = one_vs_one)
    y_classif = get_classification(allDist, np.unique(y_train))
    dic["LSVM"] = evaluate_performances(X_train, y_train, 
                                         X_test, y_test, 
                                         y_classif, 
                                         allDist, verbrose=verbrose)
    
    if verbrose:
        dic["LSVM"]["plot"]
        
    return dic
