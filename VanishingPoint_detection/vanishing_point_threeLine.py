
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters


def slopIntercept(x, y):
    slope = (y[1] - y[0])/(x[1] - x[0])
    intercept = y[1] - slope*(x[1])
    return [slope,intercept]

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def vPoint(img):
	n1 = input('How many line want to select: ')
	n = 2*n1
	lines = []
	plt.imshow(img)
	p = plt.ginput(n)

	for i in range(0, n, 4):
		plt.plot([p[i][0], p[i+1][0]], [p[i][1], p[i+1][1]], marker = 'o')
		plt.plot([p[i+2][0], p[i+3][0]], [p[i+2][1], p[i+3][1]], marker = 'o')

   	for j in range(0, n, 6):
   		for i in range(j, j+6, 2):
   			x = [p[i][0], p[i+1][0]]
   			y = [p[i][1], p[i+1][1]]
   			lines.append(slopIntercept(x, y))

   	list_vp = []
   	for j in range(0, n1, 3):
   		for i in range(j, j+3, 3):
   			ang1 = math.degrees(math.atan((lines[0][0] - lines[1][0])/(1+ lines[1][0]*lines[0][0])))
   			ang2 = math.degrees(math.atan((lines[1][0] - lines[2][0])/(1+ lines[2][0]*lines[1][0])))
   			if ang1 > 90:
   				ang1 = 180 - ang1
   			if ang2 > 90:
   				ang2 = 180 - ang2
   			if ang1 < ang2:
   				x = (lines[i][1] - lines[i+1][1])/(lines[i+1][0] - lines[i][0])
   				y = lines[i+1][0]*x + lines[i+1][1]
   				vp = (x, y)
        		#vp = line_intersection((p[i], p[i+1]), (p[i+2], p[i+3]))
        		list_vp.append(vp)
    		else:
        	   	x = (lines[i+2][1] - lines[i+1][1])/(lines[i+1][0] - lines[i+2][0])
        	   	y = lines[i+1][0]*x + lines[i+1][1]
        	   	vp = (x, y)
        		#vp = line_intersection((p[i+2], p[i+3]), (p[i+4], p[i+5]))
        		list_vp.append(vp)
        		

	plt.plot([list_vp[0][0], list_vp[1][0]], [list_vp[0][1], list_vp[1][1]], marker = 'o')
	print('The VPs coorinate are: ', list_vp)
	x1, y1 = list_vp[0]
	x2, y2 = list_vp[1]
	a, b = (y1 - y2), (x1 - x2)
	slope = a/b
	intercept = y1 - slope*(x1)
	print("The equataion is:", a,"x - ",b,"y + ",intercept*b," = 0")
	plt.show()

def main():
	filename = 'img1.jpg'
	im = Image.open(filename, 'r')
	img = np.float32(im)

	vPoint(im)


if __name__== "__main__":
	main()
