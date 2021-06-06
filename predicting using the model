
#this is supposed to predict using the model and it does, but its really bad. probably something wrong im doing here but ill fix it 

import cv2
import tensorflow as tf

CATEGORIES = ['Apple___Apple_scab', 'Apple___Black_rot', 
'Apple___Cedar_apple_rust', 'Apple___healthy', 
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

def prepare(filepath):
	IMG_SIZE = 100
	img_arr = cv2.imread(filepath)
	img_arr = cv2.resize(img_arr, (IMG_SIZE,IMG_SIZE))
	return img_arr.reshape(1,IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare(r'filepath')])
print(prediction)
print(CATEGORIES[int(prediction[0][0])])
