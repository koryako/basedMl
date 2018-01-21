from numpy import *
import operator

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    fr=open(fileName)
    dataMat=[];labelMat=[]
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print "error"
        return
    w=xTx.I*(xMat.T*yMat)
    return w


    


    

   

       


