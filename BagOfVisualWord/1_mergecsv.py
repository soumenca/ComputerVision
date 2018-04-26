import glob
import csv
index = 0
with open('trainData.csv', 'a') as singleFile:
	for i in range(1, 1888 + 1):
		path = '/home/soumen/Desktop/cv_new1/A2_Data_CV/train_sift_features'+ '/' + str(i) + '_train_sift' + '.csv'
		print(path)
		for csvFile in glob.glob(path):
			index = index + 1
			for line in open(csvFile, 'r'):
	    			line1 = str(i) +","+line
            			singleFile.write(line1)
	    

print "Number of CSV file Read is {}".format(index)
