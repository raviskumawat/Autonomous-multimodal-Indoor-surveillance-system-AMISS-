#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8

from keras import applications
import cv2
import re
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from keras.utils import to_categorical


# In[2]:


img_width=256
img_height=256


def load_small_img_dataset(path=os.getcwd()):
    data=[]
    labels=[]
    for image in os.listdir(path):
        if image.endswith(('.jpg','.jpeg','.png','bmp'),0,len(image)):
            label=re.findall('^(.*)_.*',image)[0]
            labels.append(label)
            pixels=cv2.imread(os.path.join(path,image))
            pixels=cv2.resize(pixels,(256,256),interpolation=cv2.INTER_CUBIC)
            #first make all images of same size using crop
            data.append(pixels)
            
    return data,labels

x,y=load_small_img_dataset('D:\projects\ATM public safety\second review\Face Recognition\Face_Dataset')



# In[3]:



persons=set(y)
num_persons=len(set(y))
categorical_mapping={}

#convert into one hot encoding
for i,name in enumerate(persons):
    #print(i,name)
    categorical_mapping[name]=i

output_d=[]

for i_ in y:
    i_=categorical_mapping[i_]
    output_d.append(i_)
output_d=to_categorical(output_d)


# In[4]:


def test_train_dev_split(input_data,output_data,train=0.7,dev=0.2,test=0.1):
    #make seed for exact results everything
    #random.sort(dataset)
    random.seed(2)
    random.shuffle(input_data)
    random.shuffle(output_data)
    split1=int(train*len(input_data))
    split2=int((train+dev)*len(input_data))
    train_input=input_data[:split1]
    dev_input=input_data[split1:split2]
    test_input=input_data[split2:]
    
    
    train_output=output_data[:split1]
    dev_output=output_data[split1:split2]
    test_output=output_data[split2:]
    
    return np.array(train_input),np.array(train_output),np.array(dev_input),np.array(dev_output),np.array(test_input),np.array(test_output)


# In[5]:


train_input,train_output,dev_input,dev_output,test_input,test_output=test_train_dev_split(x,output_d)


# In[6]:


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))


# In[7]:


train_output.shape


# In[8]:


model.summary()


# In[10]:


for layer in model.layers[:5]:
    layer.trainable = False


# In[16]:


#Adding custom Layers 
x = model.output
x = Flatten()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(216, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(num_persons, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

# compile the model 
model_final.compile(optimizer = optimizers.SGD(lr=0.001, momentum=0.9),loss = "categorical_crossentropy", metrics=["accuracy"])


# In[17]:


model_final.output_shape


# In[18]:


train_output.shape


# In[19]:


# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

'''
# Train the model 
model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early])
'''

model_final.fit(train_input,train_output,batch_size=5,nb_epoch=10,validation_data=(dev_input,dev_output),callbacks = [checkpoint, early])


# In[20]:


score=model_final.evaluate(test_input,test_output)
#print(score)
print("[INFO] Loss:{0}   Accuracy:{1}".format(score[0],score[1]))


# In[21]:


test_predictions=model_final.predict(test_input)


# In[22]:


oneHot2Name={}

for i in categorical_mapping.keys():
    oneHot2Name[categorical_mapping[i]]=i
oneHot2Name


# In[26]:


c=0
for i in test_predictions:
    cv2.imshow(str(oneHot2Name[np.argmax(i)])+str(c)+'.jpg',test_input[c])
    c+=1
    
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[144]:


get_ipython().run_line_magic('reset', '')


# In[162]:


1/14


# In[ ]:




