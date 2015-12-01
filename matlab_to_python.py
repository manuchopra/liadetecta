# python

from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
import h5py
import numpy

######## Short Explanation & Sizes ##########
# X: our main feature vector with size 742X17175
# 	each row is an image
# 	columns are ordered such:
# 		for each chnannel
# 			276 values of 1st channel, mean(176 val fist channels)
# 		at the end value 17175 is whether subject saw the image before or not
# 
# y: lie/no lie values for all iamges 742X1
# 
# means: a smaller feature matrix only with the means for each channels 742X62
# 
# ben experiment size: (600, 17175)
# 
# 


# get Matrix Dimenstions
def getNumRowsCols(m):
	numRows = len(m[:,0])
	numCols = len(m[0,:])
	return numRows, numCols

# X is currently imported as an array of arrays where each array represents a feature instead of a sample
# we need tp reconstruct X
def transformX(Xi):
	X = np.zeros([len(Xi[0]), len(Xi)])
	# 17175
	for i in range(len(Xi)):
		# 742 
		for j in range(len(Xi[0])):
			X[j][i] = Xi[i][j]
	return X

def getData():
	with h5py.File('eeg_X.h5', 'r') as f:
	    X = f['/extractor'].value
	with h5py.File('eeg_y.h5', 'r') as f:
	    y = f['/extractor'].value
	with h5py.File('eeg_means.h5', 'r') as f:
	    means = f['/extractor'].value
	with h5py.File('eeg_X_ben.h5', 'r') as f:
	    X_ben = f['/extractor'].value
	with h5py.File('eeg_y_ben.h5', 'r') as f:
	    y_ben = f['/extractor'].value
	with h5py.File('eeg_means_ben.h5', 'r') as f:
	    means_ben = f['/extractor'].value
	Xnew = transformX(X)
	X_ben_new = transformX(X_ben)
	# print len(Xnew), len(X_ben_new)
	Xcombined = np.concatenate((Xnew, X_ben_new), axis=0)
	# print len(Xcombined), len(Xcombined[0])
	ycombined = np.concatenate((y, y_ben), axis=0)
	# print len(ycombined)
	
	# All our samples and data. X(1342, 17175)
	return Xcombined, ycombined

# append all ben to initial


# have final X and y


def featureSelection(X, y):
	# VarianceThreshold(threshold=(.8 * (1 - .8))) doesn't work
	# ValueError: No feature in X meets the variance threshold 0.16000
	# print X.shape
	X_new = SelectKBest(f_classif, k=100).fit_transform(X, y)
	# print len(X_new), len(X_new[0]), X_new[0]
	# print X_new.shape
	return X_new

def main(X, y):
	numSamples, numFeatures = getNumRowsCols(X)
	X = featureSelection(X, y)

X, y = getData()
main(X, y)

