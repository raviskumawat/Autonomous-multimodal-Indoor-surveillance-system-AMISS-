#[Navin_Kumar_Manaswi]_Deep_Learning_with_Applicati(z-lib.org).pdf
# Multi face tracking :  https://www.guidodiepen.nl/2017/02/tracking-multiple-faces/

import cv2
import dlib
import os

saved_faces=[]


def extract_save_faces(img_,trackers):
    #global trackers
    global saved_faces
    save_path=os.path.join(os.getcwd(),"Extracted Faces")
    print("save_path:  ",save_path)
    for fid in trackers.keys():
        if fid not in saved_faces:
            tracked_position=trackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            print("Saving Face")
            #print(img_[t_x:t_x+t_w][t_y:t_y+t_h])
            #print("Saving to:  ",os.path.join(save_path,'{0}.jpg'.format(len(saved_faces)+1)))
            cv2.imwrite(os.path.join(save_path,'{0}.jpg'.format(len(os.listdir('Extracted Faces'))+1)),img_[ t_y:t_y+t_h,t_x:t_x+t_w ])
            saved_faces.append(fid)


def tracker_exist(x,y,w,h,trackers):
    #global trackers
    for fid in trackers.keys():
        tracked_position=trackers[fid].get_position()

        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        t_center_x= t_x + 0.5*t_w
        t_center_y= t_y+ 0.5*t_h

        #check if the centerpoint of the face is within the 
        #rectangleof a tracker region. Also, the centerpoint
        #of the tracker region must be within the region 
        #detected as a face. If both of these conditions hold
        #we have a match
        center_x= x+ 0.5*w
        center_y= y+ 0.5*h

        if (x<=t_center_x <= (x+w)) and (y <=t_center_y <=(y+h)) and (t_x <=center_x <= (t_x+t_w)) and (t_y <= center_y <=(t_y+t_h)) :
            return True

    return False 




def delete_trackers(img,face_count,trackers):
    #global trackers
    #global face_count
    fidsToDelete=[]
    for fid in trackers.keys():
        track_quality=trackers[fid].update(img)

        if track_quality<9:
            fidsToDelete.append(fid)

    for fid in fidsToDelete:
        print("Removing tracker " + str(fid) + " from list of trackers")
        #trackers.pop(fid,None)
        del trackers[fid]
        #face_count-=1    # as decrease the count and then there may be duplicate entries
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyecascade=cv2.CascadeClassifier('haarcascade_eye.xml')

for video_name in os.listdir('Videos'):
    print("Video Name: ",video_name)
    vs=cv2.VideoCapture('Videos/'+video_name)
    #vs=cv2.VideoCapture(0)
    face_count=0
    eye_count=0
    trackers={}

    while True:
        rc,img=vs.read()
        #img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        #print(img)
        orig_img=img.copy()
        '''for i in range(0,3):
            rc,img=vs.read()
            continue'''
        #img = cv2.rotateImage(img, 90)
        #print(img)
        #cv2.imshow('Detector',img)
        gray_img=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)

        delete_trackers(img,face_count,trackers)
        for fid in trackers.keys():
            tracked_position =  trackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            cv2.rectangle(img, (t_x, t_y),
                                (t_x + t_w , t_y + t_h),
                                (255,0,0) ,1)
            cv2.putText(img,"Tracking {0}".format(fid),(t_x+5,t_y+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)


        faces=facecascade.detectMultiScale(gray_img,1.3,5)
        eyes=eyecascade.detectMultiScale(gray_img,1.3,5)

        x=0
        y=0
        w=0
        h=0
        max_area=0

        for (_x,_y,_w,_h) in faces:
            if _w*_h>max_area:
                x=_x
                y=_y
                h=_h
                w=_w
                max_area=_w*_h
                if not tracker_exist(x,y,h,w,trackers):
                    t=dlib.correlation_tracker()
                    t.start_track(img,dlib.rectangle(int(x-10),int(y-20),int(x+w+10),int(y+h+20)))
                    cv2.putText(img,"Detecting {0}".format(face_count),(x+5,y+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                    cv2.rectangle(img, (x-10,y-20), (x+w+10, y+h+20), (0,0,255),2)
                    trackers[face_count]=t
                    face_count+=1
        
        x1=0
        y1=0
        w1=0
        h1=0
        max_area1=0

        for (_x,_y,_w,_h) in eyes:
            if _w*_h>max_area1:
                x1=_x
                y1=_y
                h1=_h
                w1=_w
                max_area1=_w*_h
            
                cv2.rectangle(img, (x1-5,y1-10), (x1+w1+5, y1+h1+10), (0,255,0),1)
            
    
                
    
    
        cv2.imshow('Detector',img)
        extract_save_faces(orig_img,trackers)
        key=cv2.waitKey(1) & 0xFF

        if key==ord('q'):
            break
    
    
    vs.release()
    cv2.destroyAllWindows()

