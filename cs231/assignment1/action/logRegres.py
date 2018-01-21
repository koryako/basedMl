from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1],[2.0,1.1],[1.1,1.0],[0,2.0],[1.3,0.1]])
    labels=['A','A','B','B','A','A','B','B']
    return group,labels

def gradAscent(arg,dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels)
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCy=500
    weights=ones((n,1))
    for k in range(maxCy):   
        h=sigmoid(dataMatrix*weights)  
        if arg=='sdg':
            error=(labelMat[k]-h)
            weights=weights+alpha*dataMatrix[k]*error  # sdg
        else:
            error=(labelMat.transpose()-h)
            weights=weights+alpha*dataMatrix.transpose()*error  # patch grad
    return weights


def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        listFormLine=line.strip().split('\t')
        dataMat.append([1.0,float(listFormLine[0]),float(listFormLine[1])])
        labelMat.append(int(listFormLine[-1]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.50
    datingDataMat,datingLabels = file2matrix('1.txt')   
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
       


