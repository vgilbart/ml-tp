import numpy as np
from matplotlib import pyplot as plt

def top(myDist, y_test, k=1, rate=True, verbrose=False):
    topk = []
    for i in myDist:
        # Outputs the index (i.e. the class) of the sorted list
        # https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
        classifPosition = sorted(range(len(myDist[i])), key=lambda k: myDist[i][k], reverse=True)
        # Test of top k: 1 if in topk, 0 if not
        truePosition = np.where(np.unique(y_test)==y_test[i])[0][0]
        if verbrose:
            print("x :", i,
                  "\ntrue classif is", truePosition,
                  "\norder of dist vector classif :", classifPosition)
        topk.append(int(truePosition in classifPosition[0:k]))
    if rate:
        topk = sum(topk)/len(myDist)
    else:
        topk = sum(topk)
    return topk

def mat_conf(myDist, y_test):
    # https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_thumb.png
    # Order in matrix is :
    #      CLASSIFICATION
    # T   1  0
    # R   0  1
    # U
    # T
    # H
    n = np.shape(np.unique(y_test))[0]
    mat = np.zeros(shape = (n,n) )
    for i in myDist:
        # Outputs the index (i.e. the class) of the truth
        truePosition = np.where(np.unique(y_test)==y_test[i])[0][0]
        # Outputs the index (i.e. the class) of the best result
        max_index = np.argmax(myDist[i])
        # Updates the confusion matrix
        mat[truePosition][max_index] += 1

    return mat

def plot_test(X_train, y_train, X_test, y_classif):
    fig = plt.figure()
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, marker = "x", s=2)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_classif, marker='s', s=20, edgecolors='k', linewidth=0.5)
    plt.close()
    return fig

def evaluate_performances(X_train, y_train, X_test, y_test, y_classif, myDist, verbrose = True):
    # Get name of classes
    myClasses = np.unique(y_test)
    # Get final classification
    c = get_classification(myDist, myClasses)
    # Create a dictionnary with performance evaluation
    dic = {}
    #print(myDist)
    dic["top1"] = top(myDist, y_test, k=1)
    dic["top2"] = top(myDist, y_test, k=2)
    dic["confusionMatrix"] = mat_conf(myDist, y_test)
    dic["plot"] = plot_test(X_train, y_train, X_test, c)
    if verbrose:
        print("Top 1 :", dic["top1"])
        print("Top 2 :", dic["top2"])
        print("\nConfusion matrix :")
        print("Order of the classes :", myClasses)
        print("Column: Classification & Row: Truth")
        print(dic["confusionMatrix"])
        dic["plot"]
        
    return dic


# https://towardsdatascience.com/cross-validation-k-fold-vs-monte-carlo-e54df2fc179b
# 5-CV : Split data in 5
# 1 is used as test and 4 as train (and do that 5 times)
# Performances are calculated as mean of performances of each split    
def CrossValidate(X, y, n_splits = 5, k_range = range(1, 5), verbrose=True):
    # Creates n_splits that will be used for index of test and train datasets
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 9)
    dic = {}
    n_matrix = np.shape(np.unique(y))[0]
    mat = np.zeros(shape = (n_matrix, n_matrix))
    for k in k_range:
        if verbrose:
            print("------------------K=%s------------------" % k)
        top1mean, top2mean, mat = 0, 0, 0
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            allRes = kppv(X_train, y_train, X_test, k=k)
            y_classif = get_classification(allRes, np.unique(y))
            
            key = str(k) + "-ppv"
            dic[key] = evaluate_performances(X_train, y_train, 
                                             X_test, y_test, 
                                             y_classif, 
                                             allRes, verbrose=False)
            top1mean += dic[key]["top1"]
            top2mean += dic[key]["top2"]
            mat += dic[key]["confusionMatrix"]
        # Make the matrix have a total of 100 in column
        mat = mat * 100 / np.sum(mat, axis=1) 
        dic[key] = {"top1mean":top1mean/n_splits, 
                  "top2mean": top2mean/n_splits, 
                  "confusionMatrixMean": mat}
        if verbrose:
            print("Mean of top1 :", dic[key]["top1mean"],
                 "\nMean of top2 :", dic[key]["top2mean"], 
                 "\nMean of confusionMatrix :\n", dic[keypply_SVM_to_data]["confusionMatrixMean"])
            
    return dic

