# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2

def outliers_index(data,thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(data.shape) == 1:
        data = data[:,None]
    median = np.median(data, axis=0)
    diff = np.sum((data - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score < thresh
# Read in and grayscale the image
# Note: in the previous example we were reading a .jpg 
# Here we read a .png and convert to 0,255 bytescale
#image = (mpimg.imread('test_images/solidWhiteRight.jpg')*255).astype('uint8')
#image = mpimg.imread('test_images/solidWhiteRight.jpg')
image = mpimg.imread('test_images/solidWhiteCurve.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
imshape = image.shape
print(imshape)
vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 25    # minimum number of votes (intersections in Hough grid cell)
min_line_length = 1 #minimum number of pixels making up a line
max_line_gap = 150    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
rows = lines.shape[0]
#theta_data = np.zeros(rows)
#rho_data = np.zeros(rows)
slope_data = np.zeros(rows)
yint_data = np.zeros(rows)

for row in range(rows):
    x1,y1,x2,y2 = lines[row,0,:]    
    slope_data[row] = (y2-y1)/(x2-x1)
    yint_data[row] = y1-slope_data[row]*x1 
    #theta_data[row] = math.atan2(y2-y1,x2-x1)
    #rho_data[row] = (x2-x1)*math.cos(theta_data[row]) + (y2-y1)*math.sin(theta_data[row])

# Sorting the list in two groups of similar slopes
gp1_range = slope_data > 0
gp1 = np.array(list(zip(slope_data[gp1_range],yint_data[gp1_range])))
gp2 = np.array(list(zip(slope_data[~gp1_range],yint_data[~gp1_range])))

# removing outliers and taking mean
gp1_index = outliers_index(gp1[:,0]) & outliers_index(gp1[:,1])
slope1_mean, yint1_mean = gp1[gp1_index,:].mean(axis=0)
gp2_index = outliers_index(gp2[:,0]) & outliers_index(gp2[:,1])
slope2_mean, yint2_mean = gp2[gp2_index,:].mean(axis=0)

# Get maximum and minimum y-value from line data
ymax = imshape[0]
ymin = 320    # Assigned as the cut off from the region of interest
x1_max = int((ymax-yint1_mean)/slope1_mean)
x1_min = int((ymin-yint1_mean)/slope1_mean)
x2_max = int((ymax-yint2_mean)/slope2_mean)
x2_min = int((ymin-yint2_mean)/slope2_mean)

# Drawing Lines on empty image
res_lines = [[[x1_max,ymax,x1_min,ymin],[x2_max,ymax,x2_min,ymin]]]
#draw_lines(line_image,res_lines,thickness=10)
#gp2_range = not(gp1_range)
#gp1_theta = theta_data[gp1_range]
#gp1_rho = rho_data[gp1_range]
#gp2_theta = theta_data[gp2_range]
#gp2_rho = rho_data[gp2_range]

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)


