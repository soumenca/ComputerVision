from scipy.ndimage import filters
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

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
    	list_vp = []
	plt.imshow(img)
	p = plt.ginput(n)
    	for i in range(0, n, 4):
            	plt.plot([p[i][0], p[i+1][0]], [p[i][1], p[i+1][1]], marker = 'o')
		plt.plot([p[i+2][0], p[i+3][0]], [p[i+2][1], p[i+3][1]], marker = 'o')
        	vp = line_intersection((p[i], p[i+1]), (p[i+2], p[i+3]))
        	print(vp)
        	list_vp.append(vp)
    		#plt.plot(vp[0], vp[1], 'ro', ls='')
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

	R = vPoint(im)



if __name__== "__main__":
	main()
