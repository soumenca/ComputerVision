from scipy.ndimage import filters
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def computeHarrisResponse(im, sigma = 3):
	""" Compute the Harris corner detector response function
	for each pixel in a graylevel image. """

	# compute derivatives
	imx = np.zeros(im.shape)
	filters.gaussian_filter(im, (sigma,sigma), (0,1), imx) # Ix = Gx * I
	imy = np.zeros(im.shape)
	filters.gaussian_filter(im, (sigma, sigma), (1,0), imy) # Iy = Gy * I

	# compute components of the Harris matrix
	Sxx = filters.gaussian_filter(imx*imx, sigma)
	Sxy = filters.gaussian_filter(imx*imy, sigma)
	Syy = filters.gaussian_filter(imy*imy, sigma)

	# compute determinant and trace
	Sdet = Sxx*Syy - Sxy*Sxy
	Str = Sxx + Syy

	# compute eigenvalue and eigenvector
	m, n = Sxx.shape
	lamda1 = np.zeros(im.shape)
	lamda2 = np.zeros(im.shape)
	for i in range(m):
		for j in range(n):
			H = np.array([[Sxx[i,j], Sxy[i,j]], [Sxy[i,j], Syy[i,j]]])
			e_vals, e_vecs = np.linalg.eig(H)
			lamda1[i,j] = e_vals[0]
			lamda2[i,j] = e_vals[1]

	
	## compute the responce
	R1 = np.minimum(lamda1, lamda2) # The Shi-Tomasi corner response
	R2 = (lamda1*lamda2) - 0.04*(lamda1+lamda2)**2 # Harris corner response
	R3 = Sdet - 0.04*(Str**2) # Another common used Harris corner response

	return (R1, R2, R3)


def getHarrisPoints(harrisim, min_dist = 5, threshold = 0.03):  ##  random.uniform(0.02, 0.04)
	""" Return corners from a Harris response image	min_dist is the minimum number 
	of pixels separating corners and image boundary. """

	# find top corner candidates above a threshold
	corner_threshold = harrisim.max() * threshold
	harrisim_t = (harrisim > corner_threshold)
	#print harrisim_t.shape

	# get coordinates of candidates
	coords = np.array(harrisim_t.nonzero()).T
	#print coords.shape

	# ...and their values
	candidate_values = [harrisim[c[0],c[1]] for c in coords]

	# sort candidates
	index = np.argsort(candidate_values)

	# store allowed point locations in array
	allowed_locations = np.zeros(harrisim.shape)
	allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
	
	# select the best points taking min_distance into account
	filtered_coords = []
	for i in index:
		if allowed_locations[coords[i,0],coords[i,1]] == 1:
			filtered_coords.append(coords[i])
			allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
			(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
	return filtered_coords


def plotHarrisPoints(image, filtered_coords):
	""" Plots corners found in image. """
	plt.figure()
	plt.gray()
	plt.imshow(image)
	plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
	plt.axis('off')
	plt.show()


def main():
	filename = 'Img3.jpg'
	img = cv2.imread(filename)
	grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	im = np.float32(grayimg)

	R = (R1, R2, R3) = computeHarrisResponse(im)
	for i in range(len(R)):
		filtered_coords = getHarrisPoints(R[i], 5)
		plotHarrisPoints(im, filtered_coords)



if __name__== "__main__":
	main()

