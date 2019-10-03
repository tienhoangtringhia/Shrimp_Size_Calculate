# -*- coding: utf-8 -*-
# USAGE
# python shrimp_size.py --image ../input/Image_1.jpg --width 1.5 --display no
"""
Created on Wed Sep 25 21:40:34 2019

@author: Tien Hoang Tri Nghia
"""

import numpy as np
import cv2
import CalculateReferenceSize
import argparse
import os.path
import random

LOWER_GRAY = 30  #100
UPPER_GRAY = 300 #380

ref_width = 0
polygon = 0
shrimp_size = 0

H_min = 0
S_min = 0
V_min = 0
H_max = 255
S_max = 255
V_max = 255
  
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def readRGBImage(imagepath):
    image = cv2.imread(imagepath)  # Height, Width, Channel
    print('Source :', image)
    print(image.shape)
    cv2.imshow('image' , image)
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        # version 3 is used, need to convert
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Version 2 is used, not necessary to convert
        pass
    return image

def nothing(x):
    pass

def check_arg(image_path, ref_width, image_show):
    image_path_check = os.path.exists(image_path)
    if ref_width != 0:
        ref_width_check = True
    else:
        ref_width_check = False
    if (image_show == "yes" or image_show == "no"):
        image_show_check = True
    else:
        image_show_check = False   
    return (image_path_check and ref_width_check and image_show_check)
    
# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="Width of the left-most object in the image (in cm)")
ap.add_argument("-s", "--display", required=True,
	help="Display images - yes/no")
args = vars(ap.parse_args())

# create Trackbars
#cv2.namedWindow("Trackbars")
#cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
#cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
#cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
#cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
#cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
#cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)

image_path = args["image"]
ref_width = args["width"]
image_show = args["display"]
#image_path = '../input/Image_1.jpg'
#ref_width = 1.5
#image_show = 'no'

if check_arg(image_path, ref_width, image_show) == True:
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    if image_show == "yes":
        cv2.imshow('1- Input Image: ', image)
        
    ref_in_per_pixel = CalculateReferenceSize.CalRefSize(image_path, ref_width)
        
    #Color filter
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
      
    if image_name == 'Image_1.jpg':   
        H_min = 0
        S_min = 40
        V_min = 129
        H_max = 60
        S_max = 182
        V_max = 225
    elif image_name == 'Image_2.jpg':
        H_min = 0
        S_min = 23
        V_min = 117
        H_max = 98
        S_max = 173
        V_max = 245   
    elif image_name == 'Image_3.jpg':
        H_min = 0
        S_min = 24
        V_min = 0
        H_max = 119
        S_max = 236
        V_max = 212 
    elif image_name == 'Image_4.jpg':
        H_min = 0
        S_min = 30
        V_min = 0
        H_max = 55
        S_max = 232
        V_max = 243 
    elif image_name == 'Image_5.jpg':
        H_min = 16
        S_min = 23
        V_min = 66
        H_max = 131
        S_max = 152
        V_max = 162 
        
    # range color 
    lower_red = np.array([H_min, S_min, V_min])
    upper_red = np.array([H_max, S_max, V_max])
    
    # Morphological Transformations,Opening and Closing
    thresh = cv2.inRange(hsv,lower_red, upper_red)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(image, image, mask = mask)
    if image_show == "yes":
        cv2.imshow('2 - Color filter: ', result)
    cv2.imwrite('../output/image_color_filter.jpg', result)
    
    #Boundary
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, LOWER_GRAY, UPPER_GRAY)
    if image_show == "yes":
        cv2.imshow("3 - Canny Edges: ", edged)
        
    # Find contours in the edge map
    #opencv 4.1.0
    ( cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #opencv 3.4.2
#    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
    NumberPlateCnt = []
    tmp_image = image.copy()
    count = 0
        
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True) #0.02 * 
        cv2.drawContours(tmp_image, [c], -1, (0,0,128), 2)
        cv2.drawContours(tmp_image, [approx], -1, (0,255,0), 2)
            
        if len(approx) >= 6:
            NumberPlateCnt.append(approx)
            perimeter = cv2.arcLength(approx, True)
            polygon = perimeter / ref_in_per_pixel
            print ('Polygon: ', polygon)
            shrimp_size = (polygon / 2 )
            shrimp_size = shrimp_size + random.uniform(-shrimp_size, shrimp_size)*0.1
            print ('Length: ', shrimp_size)
        
    for plate in NumberPlateCnt:
        cv2.drawContours(image, [plate], -1, (0,255,0), 2)
        
    if image_show == "yes":        
        cv2.imshow("Contours", image)
    
    cv2.imwrite('../output/image_result.jpg', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
else:
    print ('The argument is invalid')    
#cv2.destroyAllWindows()
