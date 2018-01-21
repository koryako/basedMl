from numpy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import knn
import linearRegres
import operator


Data,label=linearRegres.loadDataSet('../cs231n/datasets/data1.txt')
w=linearRegres.standRegres(Data,label)

x=mat(Data)
yl=mat(label)
y=x*w

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(x[:,1].flatten().A[0],yl.T[:,0].flatten().A[0])
plt.savefig("t.jpg")
e=corrcoef(y.T,yl)
print e

#group,label=knn.createDataSet()
#group,label=logRegres.loadDataSet()
#arg='sdg'
#print logRegres.gradAscent(arg,group,label)


#result=knn.classify0([20920,7.326976,0.953952],group,label,5)
#print result




