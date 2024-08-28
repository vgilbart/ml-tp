import numpy as np
from matplotlib import pyplot as plt

def get_means(X, y):
    mk = {}
    for k in np.unique(y):
        wk = X[np.where(y==k),]
        mk[k] = np.mean(wk, axis=1) 
    return mk     

def dist_eucl(x, y): 
    # x is a (1, 2) matrix, each column is a dimension, 
    # y is a (1, 2) matrix, each column is a dimension
    calc = np.subtract(x, y)
    d = calc@np.transpose(calc)
    return d

def dist_mahalanobis_same_cov(x, y, matk): 
    # x is a (1, 2) matrix, each column is a dimension of our data, 
    # y is a (1, 2) matrix, each column is a dimension of our data
    # matk is a (2, 2) matrix
    calc = np.subtract(x, y)
    d = calc@np.linalg.inv(matk)@np.transpose(calc)
    return d

def dist_mahalanobis(x, y, matk, p): 
    # x is a (1, 2) matrix, each column is a dimension of our data, 
    # y is a (1, 2) matrix, each column is a dimension of our data
    # matk is a (2, 2) matrix
    # p is a probability
    calc = np.subtract(x, y)
    d = calc@np.linalg.inv(matk)@np.transpose(calc) + (np.log(np.linalg.det(matk)) - 2*np.log(p))
    return d

def get_matk(X_train, mk, n):
    sumk = np.zeros(shape = (np.shape(mk)[1],np.shape(mk)[1]))
    # Calculates sumi(xi-mk)(xi-mk)t
    for i in X_train:
        calc = np.subtract(i, mk)
        d = np.transpose(calc)@calc
        sumk += d
    matk = 1/(n-1) * sumk
    return matk

def convert_estim_gauss(allDist):
    allDist_conv = {}
    cpt=0
    for i in allDist:
        mySum = sum(np.exp(-np.array(allDist[i])))
        mkDist = []
        for j in allDist[i]:
            myVal = np.exp(-j)/mySum
            mkDist.append(myVal)
        
        allDist_conv[cpt] = mkDist
        cpt+=1
    return allDist_conv

def estim_gauss(X_train, y_train, X_test, dist = 'euclidian', into_score=True):
    # 1. Create a mean point for each class k
    mk = get_means(X_train, y_train)
    # 2. Calculate distance di between each point in X_test to each mk
    allDist = {}
    cpt=0
    p = np.unique(np.array(y_train), return_counts=True)[1]/np.shape(y_train)[0]
    for i in X_test:
        mkDist = []
        cpt1 = 0
        for j in np.unique(y_train):
            if dist=='euclidian':
                thisDist = float(dist_eucl(i, mk[j])[0])
                mkDist.append(thisDist)
            if dist=='mahalanobis':
                myp = p[cpt1]
                n = np.shape(X_train[y_train==j])[0]
                matk = get_matk(X_train[y_train==j], mk[j], n)
                thisDist = float(dist_mahalanobis(i, mk[j], matk, myp))
                mkDist.append(thisDist)
            cpt1 += 1
        allDist[cpt] = mkDist
        cpt+=1
    if into_score:
        allDist = convert_estim_gauss(allDist)
    return allDist

def apply_estim_gauss_to_data(data=1, euclidian = True, mahalanobis = True, verbrose = True):
    dic={}
    if verbrose:
        print("Applying to data", data, "\n")
    filename = "Archive/data_tp" + str(data)
    X_train, y_train = read_file(filename + '_app.txt')
    X_test, y_test = read_file(filename+'_dec.txt')
    
    if euclidian:
        if verbrose:
            print("### Euclidian :")
        allDist = estim_gauss(X_train, y_train, X_test)
        y_classif = get_classification(allDist, np.unique(y_train))
        dic["estim_gauss_euclidian"] = evaluate_performances(X_train, y_train, X_test, y_test, y_classif, allDist, verbrose=verbrose)
        if verbrose :  
            dic["estim_gauss_euclidian"]["plot"]

            

    if mahalanobis:
        if verbrose:
            print("\n### Mahalanobis :")
        allDist = estim_gauss(X_train, y_train, X_test, dist='mahalanobis')
        y_classif = get_classification(allDist, np.unique(y_train))
        dic["estim_gauss_mahalanobis"] = evaluate_performances(X_train, y_train, X_test, y_test, y_classif, allDist, verbrose=verbrose)
        if verbrose:
            dic["estim_gauss_mahalanobis"]["plot"]
            
    return dic
