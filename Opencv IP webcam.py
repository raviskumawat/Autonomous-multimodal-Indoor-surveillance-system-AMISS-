import urllib.request
import  urllib
import cv2
import numpy as np
#from matplotlib import *
#from skimage import io

#from matplotlib import *
url='http://10.160.30.45:8080/shot.jpg' #my webcam is not working ,so i am using mobile camera
imgr=urllib.request.urlopen(url)
imgnp=np.array(bytearray(imgr.read()),dtype=np.uint8)
im=0
while im in range(10):
    imgr=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(imgr.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
    cv2.imshow('video',img)
    im=im+1
    if ord('q')==cv2.waitKey(10):
        exit(0)

gray_image = cv2.cvtColor(imgnp, cv2.COLOR_BGR2GRAY)
hsvimg=colors.rgb_to_hsv(imgnp)

cv2.imshow(grayimg)
cv2.imshow(hsvimg)
