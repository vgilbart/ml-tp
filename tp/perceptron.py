import numpy as np
from matplotlib import pyplot as plt

def dimTransformation_for_perceptron(X_train, y_train):
    # Instead of searching in dim Rd, we search in dim Rd+1
    # For this we apply a transformation on X_train:
    mySize = np.shape(X_train) 
    # mySize[0] is nb of points; mySize[1] is dim of points
    new_Xtrain = np.zeros( shape = (mySize[0], mySize[1]+1))
    for i in range(0, mySize[0]): 
        if y_train[i] == y_train[0]:
            new_Xtrain[i] = np.append(X_train[i], [1])
        else :
            new_Xtrain[i] = np.append(-X_train[i], [-1])
    return new_Xtrain

def Rosenblatt_perceptron(X_train, y_train, error = 0, maxIter = 50):
    mySize = np.shape(X_train) 
    # mySize[0] is nb of points; mySize[1] is dim of points
    new_X_train = dimTransformation_for_perceptron(X_train, y_train)
    # 1. Init: choose a random vector a
    a = np.zeros(mySize[1]+1)
    bestA, bestIsCorr = a, 0
    isCorr = np.zeros(mySize[0])
    myIter = 0
    # isCorr will contain a 1 if the point is correctly classified
    # we allow a certain number of error
    # and a certain number of iterations (= epochs)
    while sum(isCorr) + error < mySize[0] :
        # 2. Consider a new point yn of T1(w1) U T2(w2)    
        for i in range(0, mySize[0]): 
            yn = new_X_train[i]
            #print(a, yn, a.transpose() @ yn)
            # 3.  If a(n-1)(t) * yn <= 0 (i.e. yn is misclassified)
                # then do: a(n) = a(n-1) + yn
            if a.transpose() @ yn <= 0 :
                a = a + yn
            # Else: a(n-1)(t) * yn > 0 (i.e. yn is classified correctly)
            else : 
                pass
        # Check if our points are well classified
        isCorr = np.zeros(mySize[0]) 
        for i in range(0, mySize[0]): 
                    yn = new_X_train[i]
                    # 3.  If a(n-1)(t) * yn <= 0 (i.e. yn is misclassified)
                        # then do: a(n) = a(n-1) + yn
                    if a.transpose() @ yn <= 0 :
                        pass
                    # Else: a(n-1)(t) * yn > 0 (i.e. yn is classified correctly)
                    else : 
                        isCorr[i] = 1  
        # 4. Go back to 2 as long as all yn points are not classified
        # (i.e. as long as isCorr is not full of 1's)
        # but keep the best a that we have found yet
        if sum(isCorr) > bestIsCorr :
            bestIsCorr = sum(isCorr)
            bestA = a
        myIter += 1
        if myIter == maxIter:
            raise  ValueError('Too many epochs, please change error or maxIter parameter to get a result.')

    return a

