import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import os
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import cv2
import glob
import gc
import pickle
import keras

def lire_images(img_dir, xdim, ydim, nmax=5000) :
   
    label = 0
    label_names = []
    X = []
    y=[]
    for dirname in os.listdir(img_dir):
        print(dirname)
        label_names.append(dirname)
        data_path = os.path.join(img_dir + "/" + dirname,'*g')
        files = glob.glob(data_path)
        n=0
        for f1 in files:
            if n>nmax : break
            img = cv2.imread(f1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = cv2.resize(img, (xdim,ydim)) 
            X.append(np.array(img)) 
            y.append(label) 
            n=n+1
        print(n,' images lues')
        label = label+1
    X = np.array(X)
    y = np.array(y)
    gc.collect() 
    return X,y, label, label_names

def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()

from google.colab import drive
drive.mount('/content/drive')

X_train,y_train,Nombre_classes,Classes = lire_images("/content/drive/MyDrive/Waste Classification/Dataset/waste_photos", 224, 224, 1000)
X_test,y_test,Nombre_classes,Classes = lire_images("/content/drive/MyDrive/Waste Classification/Dataset/waste_photos2", 224, 224, 1000)

X_train.shape

X_test.shape

plt.figure(figsize=(10,20))
for i in range(0,20) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(X_test[i])
    plt.title(Classes[int(y_test[i])])

plt.figure(figsize=(10,20))
for i in range(0,20) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(X_test[i+1001])
    plt.title(Classes[int(y_test[i+1001])])

X_train = X_train / 224
X_test = X_test / 224

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_test.shape

model = Sequential()
 model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
 model.add(Flatten())
 model.add(Dense(Nombre_classes, activation='softmax'))
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train, 
                  y_train, 
                  validation_data=(X_test, y_test), 
                  epochs=10, 
                  batch_size=256, 
                  verbose=1)
model.summary()

scores = model.evaluate(X_test, y_test, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))

plot_scores(train)

model = Sequential()
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(Nombre_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train,
                  y_train,
                  validation_data = (X_test, y_test),
                  epochs = 10,
                  batch_size = 256,
                  verbose = 1)

model.summary()

predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))

plot_scores(train)

model.save("model.h5")

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import warnings
warnings.filterwarnings("ignore")



# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# # Replace this with the path to your image
image = Image.open('/content/drive/MyDrive/peas.jpg')
type(image)
# #resize the image to a 224x224 with the same strategy as in TM2:
# #resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

# #turn the image into a numpy array
image_array = np.asarray(image)
# # Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# # Load the image into the array
data[0] = normalized_image_array

# # run the inference
prediction = model.predict(data)
def waste(pred):
  if(pred==1):
    print("Recycle")
  else:
    print("Organic")
waste(np.argmax(prediction))

import pickle

filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))

loaded_model=pickle.load(open('trained_model.sav','rb'))

!pip install gradio

import gradio as gr

image = gr.inputs.Image(shape=(224,224))

label = gr.outputs.Label(num_top_classes=2)

def predict_input_image(img):
  img_4d=img.reshape(-1,224,224,3)
  prediction=model.predict(img_4d)[0]
  return {classes_x[i]: float(prediction[i]) for i in range(2)}

image = gr.inputs.Image(shape=(224,224))

label = gr.outputs.Label(num_top_classes=2)

gr.Interface(fn=predict_input_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')

import tensorflow as tf
print(tf.__version__)