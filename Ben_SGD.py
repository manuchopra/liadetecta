# python

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
# means: a smaller feature matrix only with the means for each channels 742X62 // there are 62 channels total


import h5py
import numpy
import csv
import time
import itertools
with h5py.File('eeg_X.h5', 'r') as f:
    X = f['/extractor'].value
with h5py.File('eeg_y.h5', 'r') as f:
    y = f['/extractor'].value
with h5py.File('eeg_means.h5', 'r') as f:
    means = f['/extractor'].value
with h5py.File('eeg_X_ben.h5', 'r') as f:
    X2 = f['/extractor'].value
with h5py.File('eeg_y_ben.h5', 'r') as f:
    y2 = f['/extractor'].value

x = numpy.array(X)
x2 = numpy.array(X2)
data = numpy.row_stack((x.T,x2.T))

answers = numpy.hstack((y,y2))
answers = answers.T

# print"answers and data:"
# print len(answers)
# print len(data)
# print len(data[0])



#switch 0 to -1
for i in range(len(answers)):
	if answers[i] == 0.0 :
		 answers[i] = -1.0

#############################
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        #return sum(d1.get(f, 0) * v for f, v in d2.items())
        return sum(d1[i]*d2[i] for i in range(len(d2)))

def increment(w, scale, data):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for i in range(len(w)):
        w[i] -=  scale * data[i]
# ####################################################
# ### SGD : stochastic gradient descent 
# numTrainingSamples = 600
# start_time = time.time()
# #Initialize the weights array to 0
# weights = [0.0 for _ in xrange(0,17174)]   #should be 17174
# stepSize = 0.025 #step size
# numIter = 4  #no iterations to begin with 

# for _ in range(numIter):
# 	for i in xrange(0,numTrainingSamples):  #should be 742
# 		margin = dotProduct(weights, data[i]) * answers[i] # y
# 		if margin < 1: 
# 			# print len(weights)
# 			# print len(data[0])
# 			increment(weights, stepSize * answers[i], data[i]) #equal to w -> w-n * dLoss

# #print weights
# print "My program took", time.time() - start_time, "to run" #20 sec
# print "numTrainingSamples", numTrainingSamples
#weights are:
#print weights	 

# # #####################################################
# # #######			check accuracy:				#########	
# # #####################################################
# correctResponses = 0
# guiltyPredictions = 0
# innocentPredictions = 0
# for i in xrange(numTrainingSamples,1342): #0
# 	result = dotProduct(weights, data[i])
# 	if result >= 1.0:
# 		result = 1.0
# 		guiltyPredictions += 1
# 	else:
# 		result = -1.0
# 		innocentPredictions += 1

# 	if answers[i] == result:
# 		correctResponses += 1
# accuracy = float(correctResponses)/ 1342

# print "My accuracy is: ",  accuracy
# print "guiltyPredictions: ",  guiltyPredictions
# print "innocentPredictions: ",  innocentPredictions


##############################################
######## builtin python methods ######
split = 600

from sklearn.svm import SVC
clf = SVC()
datatr = data[1:split,1:17174]
answerstr = answers[1:split]

clf.fit(datatr, answerstr) 

datapred = data[split:1342,1:17174]
answerspred = answers[split:1342]

print(clf.predict(datapred))
print answerspred

print clf.score(datapred,answerspred)







