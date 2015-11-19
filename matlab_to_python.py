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
# means: a smaller feature matrix only with the means for each channels 742X62


import h5py
import numpy
with h5py.File('eeg_X.h5', 'r') as f:
    X = f['/extractor'].value
with h5py.File('eeg_y.h5', 'r') as f:
    y = f['/extractor'].value
with h5py.File('eeg_means.h5', 'r') as f:
    means = f['/extractor'].value
print X
print y
print means
