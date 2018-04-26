import pandas as pd
import numpy as np
import csv

number_claster = 64
input_data = pd.read_csv('train_image_claster_map_64.csv', header = None)
input_data = input_data.values
print("Shape of the input data is {}".format(input_data.shape))

label_data = pd.read_csv('train_labels.csv', header = None)
label_data = label_data.values
print("Shape of the label data is {}".format(label_data.shape))

temp_feature = np.zeros(number_claster, dtype = int)
index = 0
temp_feature[int(input_data[0, 1:2])] = temp_feature[int(input_data[0, 1:2])] + 1
for i in range(1, input_data.shape[0]):
	if(input_data[(i-1), 0:1] == input_data[i, 0:1]):
		temp_feature[int(input_data[i, 1:2])] = temp_feature[int(input_data[i, 1:2])] + 1
	else:
		temp_feature = temp_feature.tolist()
		temp_feature.append(int(label_data[0, index]))
		with open("train_feature_64.csv", "a") as fp:
    			wr = csv.writer(fp, dialect='excel')
    			wr.writerow(temp_feature)		

		temp_feature = []		
		temp_feature = np.zeros(number_claster, dtype = int)
		temp_feature[int(input_data[i, 1:2])] = temp_feature[int(input_data[i, 1:2])] + 1
		index = index + 1

temp_feature = temp_feature.tolist()
temp_feature.append(int(label_data[0, index]))
with open("train_feature_64.csv", "a") as fp:
	wr = csv.writer(fp, dialect='excel')
	wr.writerow(temp_feature)
temp_feature = []



