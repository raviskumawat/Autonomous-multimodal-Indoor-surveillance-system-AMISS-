from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import argparse
import cv2
from keras.models import model_from_json

arg=argparse.ArgumentParser()
arg.add_argument('-i','--image',help='Path to image to be classified',nargs='+')
args=vars(arg.parse_args())



#model = MobileNetV2(weights='imagenet')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("savedWeightsMobileNet.h5")
print("Loaded model from disk")
model=loaded_model

img_path = args.get('image',None)
img = image.load_img(img_path[0], target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
preds=decode_predictions(preds, top=3)[0]
print('Predicted:', preds[0][1])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
'''
model.save('MobilenetModel.h5')
jsonmodel=model.to_json()
model.save_weights('savedWeightsMobileNet.h5')

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
print("Saved model to disk")'''

test_img=cv2.imread(img_path[0])
cv2.putText(test_img, "{0} / {1} / {2}".format(preds[0][1],preds[1][1],preds[2][1]) , (20,20) ,cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0) , 2)
cv2.imshow("Classification",test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()