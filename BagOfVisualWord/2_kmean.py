from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import csv
from scipy.spatial import distance

nc = 128

# Importing the dataset
train_data = pd.read_csv('trainData.csv', header = None)
print("Shape of the training data is {}".format(train_data.shape))
train_image_id = train_data.iloc[:,0:1].values
train_image_id = train_image_id.reshape(max(train_image_id.shape),)


test_data = pd.read_csv('testData.csv', header = None)
print("Shape of the testing data is {}".format(test_data.shape))
test_image_id = test_data.iloc[:,0:1].values
test_image_id = test_image_id.reshape(max(test_image_id.shape),)
#np.savetxt("test_image_id.csv", image_id, delimiter=",")

def kmean(train_X, test_X, nc):
	# Number of clusters
	kmeans = KMeans(n_clusters=nc)
	# Fitting the input data
	kmeans = kmeans.fit(train_X)
	# Getting the cluster labels
	train_labels = kmeans.predict(train_X)
	test_labels = kmeans.predict(test_X)

	#labels = np.add(labels, 1)
	#np.savetxt("test_image_label.csv", labels, delimiter=",")
	# Centroid values
	#centroids = kmeans.cluster_centers_
	
	#myFile = open('train_centroid_data_8.csv', 'w')
	#with myFile:
	#	writer = csv.writer(myFile)
 	#	writer.writerows(centroids)
	return train_labels, test_labels

train_image_label, test_image_label = kmean(train_data.iloc[:,5:], test_data.iloc[:,5:], nc)
train_img_clster_map = np.column_stack((train_image_id, train_image_label))
np.savetxt("train_image_claster_map_128.csv", train_img_clster_map, delimiter=",")

test_img_clster_map = np.column_stack((test_image_id, test_image_label))
np.savetxt("test_image_claster_map_128.csv", test_img_clster_map, delimiter=",")


