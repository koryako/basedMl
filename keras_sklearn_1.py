

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def create_model(neurons=1024):
    # create model
    model = Sequential()
    model.add(Dense(neurons,input_shape = (784,),activation="relu"))
    optimizer = SGD(lr=0.2)
    model.add(Dense(10,activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=1, batch_size=128)

neurons = [20,30,50,100,200]
epochs = [4,6,8]

param_grid = dict(neurons=neurons, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


plt.errorbar(neurons,means[:5],stds[:5],lw=3,alpha=0.7,color='b')
plt.plot(neurons,means[:5],'ok',ms=8)

plt.errorbar(neurons,means[5:10],stds[5:10],lw=3,alpha=0.7,color='g')
plt.plot(neurons,means[5:10],'ok',ms=8)

plt.errorbar(neurons,means[10:15],stds[10:15],lw=3,alpha=0.7,color='r')
plt.plot(neurons,means[10:15],'ok',ms=8)

plt.xlabel('Neurons',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.title('Acc vs. # neurons')
plt.show()

"""
import numpy as np
from sklearn import datasets, linear_model

x_data = np.meshgrid(neurons,epochs)
x = np.zeros([15,2])
for ii,(n,e) in enumerate(zip(x_data[0].flatten(),x_data[1].flatten())):
    x[ii,:] = [n,e]
    
x = x.astype(np.float32)
y = means.astype(np.float32)


regr = linear_model.LinearRegression()
regr.fit(x,y)
print('Coefficients: \n', regr.coef_)
plt.plot(y,regr.predict(x),'o',ms=10,alpha=0.7)
plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
plt.xlabel('Real means',fontsize=20)
plt.ylabel('Model prediction',fontsize=20)
plt.show()
"""