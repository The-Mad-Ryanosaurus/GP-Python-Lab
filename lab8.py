import cv2
import numpy as np
from matplotlib import pyplot as plt

# print('Hello World')

# Image Declared
img = cv2.imread('ATU.jpg',)
cv2.imshow('Original', img)
cv2.waitKey(0)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gray Image Declared
cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)

# Blurred Images Declared
imgOut1 = cv2.GaussianBlur(gray_image,(3, 3),0)
imgOut2 = cv2.GaussianBlur(gray_image,(13, 13),0)

# Sobel Image
sobelHorizontal = cv2.Sobel(imgOut1,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(imgOut1,cv2.CV_64F,0,1,ksize=5) # y dir

# Canny Image
canny = cv2.Canny(imgOut1,100, 200)

# Rows and Columns for Display
# When plotting points rows never change but column does. If ncol = 2 then ncol 3 puts image on new row
nrows = 4
ncols = 2


plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,3),plt.imshow(imgOut1, cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,4),plt.imshow(imgOut2, cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,5),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,6),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel Vertical'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,7),plt.imshow(sobelHorizontal + sobelVertical, cmap = 'gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,8),plt.imshow(canny, cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])



plt.show()



