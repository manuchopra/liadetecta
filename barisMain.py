# python

from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from math import ceil
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn import decomposition
import json
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

######### Cdde from Baris CS229 ###############

def split_dataset(X, Y, test_fraction):
	num_rows = len(X[:,0])
	num_cols = len(X[0,:])

	num_test = ceil(num_rows * test_fraction)
	num_train = num_rows - num_test

	shuffled_train = np.empty((num_train, num_cols))
	shuffled_test = np.empty((num_test, num_cols))

	permutation = np.random.permutation(num_rows)
	all_indeces = set(np.arange(0,num_rows))
	test_indeces = set(permutation[:num_test])
	train_indeces = all_indeces - test_indeces

	train_data = X[list(train_indeces),:]
	test_data = X[list(test_indeces),:]

	train_y = Y[list(train_indeces)]
	test_y = Y[list(test_indeces)]

	return train_data, test_data, train_y, test_y

def cross_validate(X, Y, model, kFold, test_fraction):
	
	accuracies = []
	true_pos_counts = []
	true_neg_counts = []
	false_pos_counts = []
	false_neg_counts = []

	# print("________________________________________")
	# print("RESULTS:")
	for k in range(kFold):
		train_data, test_data, train_y, test_y = split_dataset(X, Y, test_fraction)
		model.fit(train_data, train_y)
		correct_count = 0
		true_pos_count = 0
		true_neg_count = 0
		false_pos_count = 0
		false_neg_count = 0
		for i in range(len(test_data)):
			pred = model.predict(test_data[i,:])
			actual = test_y[i]
			# print("Pred: ", pred, "Actual: ", actual)
			if pred == actual:
				correct_count += 1
				if pred == 0:
					true_pos_count += 1
				else:
					true_neg_count += 1
			else:
				if pred == 0:
					false_pos_count += 1
				else:
					false_neg_count += 1


		test_acc = float(correct_count) / float(len(test_data))
		num_pos = len([1 for i in range(len(test_data)) if test_y[i] == 0])
		num_neg = len([1 for i in range(len(test_data)) if test_y[i] == 1])

		accuracies.append(test_acc)
		true_pos_counts.append((float(true_pos_count) + 1) / (num_pos + 1))
		true_neg_counts.append(float(true_neg_count + 1) / (num_neg + 1))
		false_pos_counts.append(float(false_pos_count + 1) / (num_neg + 1))
		false_neg_counts.append(float(false_neg_count + 1) / (num_pos + 1))


	avg_true_pos_rate = float(sum(true_pos_counts)) / kFold
	avg_true_neg_rate = float(sum(true_neg_counts)) / kFold
	avg_false_neg_rate = float(sum(false_neg_counts)) / kFold
	avg_false_pos_rate = float(sum(false_pos_counts)) / kFold
	avg_acc = float(sum(accuracies)) / kFold


	# print("________________________________________")
	# print("OVERALL ACCURACY: " + str(avg_acc))
	# print("TRUE NEG RATE: " + str(avg_true_neg_rate))
	# print("TRUE POS RATE: " + str(avg_true_pos_rate))
	# print("FALSE POS RATE: " + str(avg_false_pos_rate))
	# print("FALSE NEG RATE: " + str(avg_false_neg_rate))
	# print("________________________________________")

        bal_acc = 0
        avg_acc = (avg_true_pos_rate + avg_true_neg_rate)/(avg_true_pos_rate + avg_false_neg_rate + avg_true_neg_rate + avg_false_pos_rate)
        bal_acc = 0.5*(avg_true_pos_rate)/(avg_true_pos_rate + avg_false_neg_rate) + 0.5*(avg_true_neg_rate)/(avg_true_neg_rate + avg_false_pos_rate)
        
	return (bal_acc, avg_acc)


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
	# append all ben to initial
	return Xcombined, ycombined




# have final X and y
def featureSelection(X, y, numFeatures):
	# VarianceThreshold(threshold=(.8 * (1 - .8))) doesn't work
	# ValueError: No feature in X meets the variance threshold 0.16000
	# print X.shape
	X_new = SelectKBest(f_classif, k=numFeatures).fit_transform(X, y)
	# print len(X_new), len(X_new[0]), X_new[0]
	# print X_new.shape
	return X_new


# def pcaTry(X):
#     pca = decomposition.PCA(n_components=3)
#     pca.fit(X)
#     X = pca.transform(X)
#     return X

def modelSelection(X, y):
	K_FOLDS = 200
	TEST_FRACTION = 0.3

	model_arr = [
				LDA(),
				DecisionTreeClassifier(max_depth=5), 
				KNeighborsClassifier(3), 
				SVC(gamma=2, C=1), 
				SVC(kernel="linear", C=0.025), 
				GaussianNB(), 
				LogisticRegression()]
	
	model_names = [
				"LDA()",
				"DecisionTreeClassifier(max_depth=5)", 
				"KNeighborsClassifier(3)", 
				"SVC(gamma=2, C=1)", 
				"SVC(kernel=linear, C=0.025)", 
				"GaussianNB()", 
				"LogisticRegression()"]

	for i, m in enumerate(model_arr):
		result = cross_validate(X, y, m, K_FOLDS, TEST_FRACTION)
		print model_names[i]
		print result

def main(X, y):
	# numSamples, numFeatures = getNumRowsCols(X)
	selectNumFeatures = 1000
	X = featureSelection(X, y, selectNumFeatures)
	# X = pcaTry(X)
	print "X: ", X.shape 
	print "Number of Features: ", selectNumFeatures
	modelSelection(X, y)
	

X, y = getData()
main(X, y)

