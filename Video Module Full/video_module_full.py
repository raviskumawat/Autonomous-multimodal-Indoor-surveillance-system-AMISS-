# [Navin_Kumar_Manaswi]_Deep_Learning_with_Applicati(z-lib.org).pdf
# Multi face tracking :  https://www.guidodiepen.nl/2017/02/tracking-multiple-faces/

#import gc
import cv2
from dlib import get_frontal_face_detector, correlation_tracker, rectangle,shape_predictor
from os import path, getcwd, listdir, walk
import numpy as np
from face_recognition import face_encodings,compare_faces,load_image_file
from imageai.Detection import ObjectDetection
#from warnings import filterwarnings
import datetime
import time
import imutils
from keras.models import load_model,Model
#import dlib
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from pickle import load,dump,loads
#from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications import inception_resnet_v2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import re

#filterwarnings("ignore")
#from FaceRecogEncodings_SVM import recognise_img_SVM
#saved_faces = []
face_detector = get_frontal_face_detector()
object_detector = ObjectDetection()
object_detector.setModelTypeAsRetinaNet()
object_detector.setModelPath('resnet50_coco_best_v2.1.0.h5')
object_detector.loadModel(detection_speed='fast')
min_area = 1000
#num_to_labels = np.load('num_to_labels.npy').item()
emotion_model = load_model('image_emotion_fer_jaffe_raf_encodings_rus_47.h5')
ultimate_label_mapping = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
    
}
num_to_labels={v:k for k,v in ultimate_label_mapping.items()}
shape_predictor_ = 'shape_predictor_68_face_landmarks.dat'
#detector = dlib.get_frontal_face_detector()
predictor = shape_predictor(shape_predictor_)
fa = FaceAligner(predictor, desiredFaceWidth=256)

def align_face(image):
    image = imutils.resize(image, width=200)
    image=image.astype(np.uint8)
    if image.ndim >2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray=image
    rects = face_detector(gray, 2)
    # loop over the face detections
    '''for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)'''
    if len(rects)>0:
        faceAligned = fa.align(image, gray, rects[0])
    else:
        faceAligned=image
    
    return faceAligned


def detect_emotion(image_):
    # image=cv2.imread(image)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    #image = cv2.cvtColor(image_.copy(), cv2.COLOR_BGR2GRAY)
    #print(image.shape)
    image=cv2.resize(image_.copy(),(256,256))
    image=align_face(image.reshape(256,256).astype(np.float32))
    encoded_image=face_encodings(image[:,:,np.newaxis].repeat(3, 2),known_face_locations=[(0,image.shape[1],image.shape[0],0)])
    encoded_image=np.array(encoded_image).ravel().reshape((1,128))
    emotion=num_to_labels[int(np.argmax(emotion_model.predict(encoded_image)))]
    '''
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    y_pred = np.argmax(emotion_model.predict(image))
    print(y_pred)
    emotion = num_to_labels[y_pred]'''
    return emotion


def detect_motion(ref_img, img):
    firstFrame = ref_img
    img = cv2.resize(img, (500, 500))
    # firstFrame=imutils.resize(firstFrame,width=500)
    if img.ndim>2:
        gray_frame = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    else:
        gray_frame=img
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    occupied = 0
    frameDelta = cv2.absdiff(firstFrame, gray_frame)
    #print("FrameDelta shape:",frameDelta.shape)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    #print("Threshold shape before diluting",thresh.shape)
    # Dilate the threshold image to fill in the holes,then find countours on the image
    thresh = cv2.dilate(thresh, None, iterations=2)
    #print("Threshold shape After diluting",thresh.shape)
    #thresh = cv2.cvtColor(thresh, cv2.CV_8UC1)

    # print(thresh)
    # print(thresh.shape)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        occupied = 1
    text = "Occupied" if occupied == 1 else "UnOccupied"
    cv2.putText(img, "Room status:  {0}".format(
        text), (15, 20+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(img, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    return occupied, img


def get_objectdetections(input_img):
    global object_detector
    '''
    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
        '''
    # plt.imshow(returned_image)
    return object_detector.detectObjectsFromImage(input_type='array', input_image=input_img, output_type='array', minimum_percentage_probability=85)


def give_embedding(img_):
    #img_ = cv2.resize(img_, (256, 256), interpolation=cv2.INTER_CUBIC)
    # extract gray img
    # gray_img=np.eye(256)
    # gray_img=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    # Create a HOG face detector using the built-in dlib class

    # print(image)
    '''face=face_detector(gray_img,1)
    if len(face)<1:
        return []'''
    # detected_faces=face
    # face=face[0]
    # face_img=cv2.rectangle(img,(detected_faces[0].left(),detected_faces[0].top()),(detected_faces[0].right(),detected_faces[0].bottom()),(0,255,0),1)
    # face_img=img[face.top():face.bottom(),face.left():face.right()]
    #img=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    if img_.ndim==3:
        img=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    else:
        img=img_
    img=cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
    img=align_face(img.astype(np.float32))
    face_embedding = np.array(face_encodings(img[:,:,np.newaxis].repeat(3, 2),known_face_locations=[(0,img.shape[1],img.shape[0],0)])).ravel()
    # Normalize???
    '''from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if len(face_embedding) > 0:
        face_embedding = face_embedding.reshape(128, 1)
        face_embedding = scaler.fit_transform(face_embedding)
    print('Face Embedding', face_embedding)
    '''
    return face_embedding


def load_face_recog_model_DNN():
    model = load_model('model_checkpoint_face_recog_aligned.h5')
    oneHot2Name = np.load('oneHot2Name.npy').item()
    # Test Image
    return model, oneHot2Name


model, oneHot2Name = load_face_recog_model_DNN()


def recognise_img_DNN(test_img):
    global oneHot2Name
    global model
    embd = give_embedding(test_img)
    #print(embd)
    if len(embd) < 1:
        return 'unknown'
    predicted = model.predict(np.array(embd).reshape(1, 128))
    face = oneHot2Name[np.argmax(predicted[0])]
    print("predicted: {0} ".format(face))
    # cv2.imshow(oneHot2Name[predicted[0]],cv2.imread(test_img))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return face

face_match_dir='D:/dataset/Image/Face Dataset custom'
def face_match(test_img):
    found=0
    test_encoding=give_embedding(test_img)
    for img_ in listdir(face_match_dir):
        img=load_image_file(face_match_dir+'/' + img_)
        current_encoding=give_embedding(img)
        #print(current_encoding.shape,test_encoding.shape)
        result=compare_faces([test_encoding], current_encoding)
        if(result[0])==True:
            face=re.search('(.*)_.*',img_).group(1)
            found=1
            print('[FACE MATCH FOUND:]....',face)
            break
    if found==0:
        face='unknown'
    return face

'''
def load_face_recog_model(train=False):
    
    with open('SVMmodel.pickle', 'rb') as file:
        model =loads(open('SVMmodel.pickle', "rb").read())
    
    oneHot2Name=np.load('oneHot2Name.npy').item()
    #Test Image
    return model,oneHot2Name
model,oneHot2Name=load_face_recog_model()


def recognise_img_SVM(test_img):
    global oneHot2Name
    global model
    embd=give_embedding(test_img)
    if len(embd)<1:
        return 'unknown'
    predicted=model.predict(np.array(embd).reshape(1,128))
    print("predicted: {0} ".format(oneHot2Name[predicted[0]]))
    #cv2.imshow(oneHot2Name[predicted[0]],cv2.imread(test_img))
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return oneHot2Name[predicted[0]]
'''


'''
def extract_save_faces(img_,trackers):

    #global trackers
    global saved_faces
    save_path= path.join( getcwd(),"Extracted Faces")
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
            #print("Saving to:  ", path.join(save_path,'{0}.jpg'.format(len(saved_faces)+1)))
            cv2.imwrite( path.join(save_path,'{0}.jpg'.format(len( listdir('Extracted Faces'))+1)),img_[ t_y:t_y+t_h,t_x:t_x+t_w ])
            saved_faces.append(fid)'''


def tracker_exist(x, y, w, h, trackers):
    #global trackers
    for fid in trackers.keys():
        tracked_position = trackers[fid][0].get_position()

        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        t_center_x = t_x + 0.5*t_w
        t_center_y = t_y + 0.5*t_h

        # check if the centerpoint of the face is within the
        # rectangleof a tracker region. Also, the centerpoint
        # of the tracker region must be within the region
        # detected as a face. If both of these conditions hold
        # we have a match
        center_x = x + 0.5*w
        center_y = y + 0.5*h

        if (x <= t_center_x <= (x+w)) and (y <= t_center_y <= (y+h)) and (t_x <= center_x <= (t_x+t_w)) and (t_y <= center_y <= (t_y+t_h)):
            return True

    return False


def delete_trackers(img, face_count, trackers):
    #global trackers
    #global face_count
    fidsToDelete = []
    for fid in trackers.keys():
        track_quality = trackers[fid][0].update(img)

        if track_quality < 9:
            fidsToDelete.append(fid)

    for fid in fidsToDelete:
        print("Removing tracker " + str(fid) + " from list of trackers")
        # trackers.pop(fid,None)
        del trackers[fid]
        # face_count-=1    # as decrease the count and then there may be duplicate entries

# Don't need Haar detector, will detect using dlibs HOG
# facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


'''
eyecascade=cv2.CascadeClassifier('haarcascade_eye.xml')
'''
#face_detector = get_frontal_face_detector()
'''
for video_name in  listdir('Videos'):
    print("Video Name: ",video_name)
    vs=cv2.VideoCapture('Videos/'+video_name)
'''


def show_img(img):
    #print('[INFO] Showing Image')
    stop = 0
    cv2.imshow('Security Feed', img)
    # to make faster ENABLE to save output data to be used as training data
    # extract_save_faces(orig_img,trackers)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        stop = 1
        cv2.destroyAllWindows()
    return stop


# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
caption_model = load_model('model_18.h5')
# load and prepare the photograph

# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = inception_resnet_v2.InceptionResNetV2(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000)
    # re-structure the model
    #model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    #image = load_img(filename, target_size=(299, 299))
    image = cv2.resize(filename,(299,299))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = inception_resnet_v2.preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = caption_model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

caption='Empty Caption'
caption_threat_check='No'
def get_caption(img):
    global caption
    global caption_threat_check
    feat = extract_features(img)
    # generate description
    caption = generate_desc(model, tokenizer,feat, max_length)
    caption_threat_check=caption_classifier()

loaded_model = load(open('threat_model_rfc', 'rb'))
vectorizer = load(open('vectorizer', 'rb'))
def caption_classifier():
    global caption
    clean_test_reviews=[]
    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(caption,True)))
    test_data_features=vectorizer.transform(clean_test_reviews)
    test_data_features=test_data_features.toarray()
    threat=loaded_model.predict(test_data_features)
    return threat

from threading import Thread

stop = 0
counter=0

while not stop:
    vs = cv2.VideoCapture(0)
    vs.set(cv2.CAP_PROP_FPS, 12)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    face_count=0
    # eye_count=0
    trackers = {}
    _, first_frame = vs.read()
    del _
    # print(first_frame)
    first_frame = cv2.resize(first_frame, (500, 500))
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)
    while True:
        rc, img = vs.read()
        '''for i in range(0,5):
            x,y=vs.read()'''
        #img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        # print(img)

        motion, img = detect_motion(first_frame, img)
        if(not motion):
            stop = show_img(img)
            if(stop):
                break
            print('[MOTION NOT DETECTED].....................')

            continue
        print('[MOTION DETECTED].....................')

        img, detections = get_objectdetections(img)
        print('[INFO] Got Objects Present in the Image')

            
        
        if counter%40==10:
            print('[Generating captions].....................')
            get_caption(img)
            #t1 =Thread(target=get_caption, args=(img,)) 
            #t1.start()
            print('[CAPTION] GENERATED...{0}'.format(caption))
            #threat_caption=caption_classifier(caption)
            print('[THREAT CLASSIFIER] ...{0}'.format(caption_threat_check))
            # orig_img=img.copy()q
        counter+=1
        
        cv2.putText(img,"Caption: {0}".format(caption), (15, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img,"Threat: {0}".format(str(caption_threat_check)), (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        '''for i in range(0,3):
            rc,img=vs.read()
            continue'''
        #img = cv2.rotateImage(img, 90)
        # print(img)
        # cv2.imshow('Detector',img)
        print('[INFO] Preprocessed Image From VideoStream')
        delete_trackers(img, face_count, trackers)

        print('[INFO] Deleted Trackers')

        
        # faces=facecascade.detectMultiScale(gray_img,1.3,5)
        '''
        eyes=eyecascade.detectMultiScale(gray_img,1.3,5)'''
        l=[d['name'] for d in detections]
        if 'person' in l:
            gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
            faces_rect = face_detector(gray_img, 1)

            print('[INFO] Detected Faces from the image')
            # detected_faces=face

            #CREATE and ADD trackers for new faces
            x = 0
            y = 0
            w = 0
            h = 0
            max_area = 0

            faces = []
            for i in faces_rect:
                faces.append((i.left(), i.top(), i.right() -i.left(), i.bottom()-i.top()))

            for (_x, _y, _w, _h) in faces:
                if _w*_h > max_area:
                    x = _x
                    y = _y
                    h = _h
                    w = _w
                    max_area = _w*_h
                    if not tracker_exist(x, y, h, w, trackers):
                        t1 = correlation_tracker()
                        t1.start_track(img,  rectangle(int(x), int(y), int(x+w), int(y+h)))
                        cv2.putText(img, "New Face detected", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(img, (x-10, y-20),(x+w+10, y+h+20), (0, 0, 255), 1)
                        #face_img=img[y-5:y+h+5, x-5:x+w+5]
                        face_img = gray_img[y-5:y+h+5, x-5:x+w+5]
                        '''personface1 = recognise_img_DNN(face_img)
                        personface2=face_match(face_img)
                        if personface1==personface2:
                            personface=personface1
                        else:
                            personface='unknown'''
                        personface = recognise_img_DNN(face_img)
                        #emotion_img = cv2.cvtColor(face_img.copy(), cv2.COLOR_BGR2GRAY)
                        emotion_img = cv2.resize(face_img, (256, 256), interpolation=cv2.INTER_CUBIC)
                        emotion = detect_emotion(emotion_img)
                        print('[ADDING NEW FACE] '+str(personface))
                        trackers[face_count] = [t1, personface, emotion]
                        face_count += 1

                        print('[INFO] Added New_faces to tracker list')

        #ADD rectangles and text to existing trackers
        for fid in trackers.keys():
            tracked_position = trackers[fid][0].get_position()
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            # face_img=cv2.rectangle(img,(detected_faces[0].left(),detected_faces[0].top()),(detected_faces[0].right(),detected_faces[0].bottom()),(0,255,0),1)
            #face_img = img[t_y-5:t_y+t_h+5, t_x-5:t_x+t_w+5]
            # face_embedding=np.array(face_recognition.face_encodings(face_img)).ravel()
            recog_face = trackers[fid][1]
            emotion = trackers[fid][2]
            cv2.rectangle(img, (t_x, t_y),
                          (t_x + t_w, t_y + t_h),
                          (255, 0, 0), 1)
            cv2.putText(img, "{0}".format(recog_face), (t_x+5, t_y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img, "{0}".format(emotion), (t_x+5, t_y+t_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        print('[INFO] Added Rectangles and Text for Existing Trackers')





        '''
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
            '''

        
        stop = show_img(img)
        '''if counter%15==0:
            t1.join()'''
        if(stop):
            break

    vs.release()
    cv2.destroyAllWindows()

