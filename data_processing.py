
import pickle 
import time 
import os 
import cv2
import random 
import matplotlib.pyplot as plt 
import numpy as np 




DIRECTORY = r'C:\Users\sahme\Desktop\Collection\Sublime Files\kaggle datasets\archive\PlantVillage Dataset (Labeled)\Color Images'
CATEGORIES = [
'Potato___Early_blight', 'Potato___Late_blight', 
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___healthy',
'Corn_(maize)___Northern_Leaf_Blight', 
'Tomato___Bacterial_spot', 'Tomato___Late_blight', 'Tomato___Septoria_leaf_spot',
'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 
'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
]

IMG_SIZE = 100

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY,category)
    label = CATEGORIES.index(category)
    print(label)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE,IMG_SIZE))
        data.append([img_arr,label])

random.shuffle(data)

X = []
Y = []

for features, labels in data:
	X.append(features)
	Y.append(labels)

X = np.array(X)
Y = np.array(Y)

X = X/255

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()


# print(X.shape) #Contains our matrix representations of all images of  leaves that are in 100x100 pixels and contain 3RGB channels
#print(Y.shape) #Contains all the index values of the Categories list i.e. if the apple leaf is rotten, healthy...

