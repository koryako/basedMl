# import the necessary packages
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", required=True,
	help="path to reference OCR-A image")
args = vars(ap.parse_args())


# define a dictionary that maps the first digit of a credit card
# number to the credit card type
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}
# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*

ref = cv2.imread(args["reference"])
ref  = imutils.resize(ref ,width=300)
ref = ref[0:150, 0:200]
refgray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
#ref = cv2.threshold(refgray, 20, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", ref)




ref  = cv2.threshold(refgray, 20, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
#cv2.imshow('343',img)
contour, hierarchy = cv2.findContours(ref,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
print(len(refCnts))




digits = {}
# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
    
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
 
	# update the digits dictionary, mapping the digit name to the ROI
    digits[i] = roi
print (len(digits))






# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
image = image[0:150, 0:200]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

img = cv2.threshold(gray, 20, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
#cv2.imshow('343',img)
contour, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print (type(contour))  
#print (type(contour[0]))  
#print (len(contour)) 

#cv2.drawContours(image ,contour,0,(0,0,255),1)
#cv2.drawContours(image,contour,1,(0,255,0),1)
#cv2.drawContours(image,contour,2,(255,255,0),1)
print (type(hierarchy))  
print (hierarchy.ndim)  
print (hierarchy[0].ndim)  
print (hierarchy.shape) 
cv2.imshow("img", image)
imgCnts = contour if imutils.is_cv2() else hierarchy
imgCnts = contours.sort_contours(imgCnts, method="left-to-right")[0]
print(len(imgCnts))
(x,y,w,h)=cv2.boundingRect(contour[1])
print(w,h)

output = []
imgdigit={}
# loop over the digit contours
for (i,c) in enumerate(imgCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print(x, y, w, h)
    if (y==56):
        cv2.rectangle(image, (x,y), (x+w,y+h), (180,255,0), 3)
        roi = image[y:(y+h), x:(x+w)]
        roi = cv2.resize(roi, (57, 88))
        roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        output.append(str(np.argmax(scores)))#imgdigit[i] = roi
print (output)
