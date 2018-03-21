import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import validation_curve
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.datasets.samples_generator import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier



def plotData(data, label):
	plt.scatter(data[:,0], data[:, 2], c=label.ravel())
	plt.show()

def crossValidationofKernel(x_train, y_train):
	modelScores = []
	for kernel in 'sigmoid','linear','rbf':
		trainsvm = svm.SVC(C=0.95, kernel=kernel, decision_function_shape='ovo')
		scores = cross_validation.cross_val_score(trainsvm, x_train, y_train.ravel(), cv=5)
		modelScores.append(scores.mean())
		print(kernel, scores)
	plt.plot(range(1,5), modelScores)
	plt.show()




path = '/Users/mengwang/Documents/MyCode/Machine Learning/cuSplit/CU64_QP27_Test_labels.data'
data = np.loadtxt(path, dtype=str, delimiter='  ')
data = data.astype(float)
#x, y = np.split(data, (2,), axis=1)
#x = x[:, :2]
#x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
x_train, y_train = np.split(data, (2,), axis=1)
print(x_train.mean(axis=0),'mean of x_train')
print(x_train.std(axis=0), 'std of x_train')
x_train = preprocessing.scale(x_train, axis=0)

#plotData(x_train, y_train)



path = '/Users/mengwang/Documents/MyCode/Machine Learning/cuSplit/CU64_QP22_Valid_labels.data'
data = np.loadtxt(path, dtype=str, delimiter='  ')
data = data.astype(float)
x_test, y_test = np.split(data, (2,), axis=1)
print(x_test.mean(axis=0), 'mean of x_test')
print(x_test.std(axis=0),  'std of x_test')
x_test = preprocessing.scale(x_test, axis=0)


#scaler = preprocessing.StandardScaler().fit(x_train)
#scaler.transform(x_train)
#scaler.transform(x_test)

#scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
#scaler.transform(x_train)
#scaler.transform(x_test)

#x_train = preprocessing.normalize(x_train, norm='l2')
#x_test = preprocessing.normalize(x_test, norm='l2')

#clf = svm.SVC(C=0.95, kernel='sigmoid', decision_function_shape='ovo', tol = 0.0001)
clf = svm.LinearSVC(penalty='l2', loss='squarehinge', dual=True, C=1, random_state=3, multi_class='crammer_singer')
clf.fit(x_train, y_train.ravel())


#print(clf.score(x_train, y_train))
y_pred_train = clf.predict(x_train)
print(accuracy_score(y_train, y_pred_train),'accuracy in train set')

y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred),'accuracy in test set') #accuracy_score only used in classification problem

print(clf.get_params())

print(clf.coef_)
print(clf.intercept_)





#lr_clf = LogisticRegression()
#sgdc_clf = SGDClassifier()
#lr_clf.fit(x_train, y_train)
#sgdc_clf.fit(x_train, y_train)

#y_pred_lr = lr_clf.predict(x_test)
#y_pred_sgd= sgdc_clf.predict(x_test)

#print(lr_clf.score(x_test, y_test), 'accuracy of LR classifier')
#print(classification_report(y_test, y_pred_lr, target_names=['non-split', 'split']))

#print(sgdc_clf.score(x_test, y_test), 'accuracy of SGD classifier')
#print(classification_report(y_test, y_pred_sgd, target_names=['non-split', 'split']))


print(clf.score(x_test, y_test), 'accuracy of SVM classifier')
print(classification_report(y_test, y_pred, target_names=['non-split', 'split']))


#print(lr_clf.coef_)
#print(lr_clf.intercept_)
