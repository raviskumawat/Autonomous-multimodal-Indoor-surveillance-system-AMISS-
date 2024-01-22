#  https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

from imutils.video import VideoStream
import cv2
import argparse
import time
import datetime
import imutils



arg=argparse.ArgumentParser()

arg.add_argument("-v",'--video',help='Path to video')
arg.add_argument('-ma','--min_area',type=int,default=800,help='minimum area to be detected')
args=vars(arg.parse_args())

if args.get('video',None) is None:
    vs=VideoStream(src=0).start()
    time.sleep(2.0)

else:
    vs=cv2.VideoCapture(args["video"])

firstFrame=None

while True:
    frame=vs.read()
    frame=frame if args.get('video',None) is None else frame[1]
    text="UnOccupied"

    frame=imutils.resize(frame,width=500)
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame=cv2.GaussianBlur(gray_frame,(21,21),0)

    if firstFrame is None:
        firstFrame=gray_frame
        continue
    frameDelta=cv2.absdiff(firstFrame,gray_frame)
    print("FrameDelta shape:",frameDelta.shape)
    thresh=cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]
    print("Threshold shape before diluting",thresh.shape)
    #Dilate the threshold image to fill in the holes,then find countours on the image
    thresh=cv2.dilate(thresh,None,iterations=2)
    print("Threshold shape After diluting",thresh.shape)
    #thresh = cv2.cvtColor(thresh, cv2.CV_8UC1)

    #print(thresh)
    print(thresh.shape)
    cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c)<args['min_area']:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        text="Occupied"

    cv2.putText(frame,"Room status:{0}".format(text),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.putText(frame,datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
    
    cv2.imshow("Security Feed ",frame)
    cv2.imshow("Threshold ",thresh)
    cv2.imshow("Frame Delta ",frameDelta)
    key=cv2.waitKey(1) & 0xFF

    if key==ord('q'):
        break
vs.stop() if args.get('video',None is None) is None else vs.release()
cv2.destroyAllWindows()