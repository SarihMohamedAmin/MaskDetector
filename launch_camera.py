import numpy as np
import cv2
import tensorflow as tf
from skimage.feature import hog 

camera = cv2.VideoCapture(0)

model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation = "sigmoid"), #softmax
        ])

checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path)

# model = tf.keras.models.load_model("/home/sarih/presentation/MyNeuralNetwork")
while True:
	check, frame = camera.read()
	"""Processing space"""
	detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
	gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces    = detector.detectMultiScale(gray, 1.1, 5)
	image_copy = frame.copy()
	for (x, y, h, w) in faces:
		""" Predictions """
		img_face = gray[x:(x+w), y:(y+h)]
		img_face = cv2.resize(img_face, (128,128))
		features = hog(img_face, orientations=8, pixels_per_cell=(16, 16),
				cells_per_block=(4, 4), visualize=False)
		pred = model.predict(features)
		if pred[0][0] >= 0.6:
			cv2.rectangle(image_copy, (x, y), (x+w, y+h),
				(0, 255, 0), 2)
			cv2.rectangle(image_copy, (x, y-30), (x+w, y),
				(0, 255, 0), -1)
			cv2.putText(image_copy, "Masked", (x,y-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
		else:
			cv2.rectangle(image_copy, (x, y), (x+w, y+h),
				(0, 0, 255), 2)
			cv2.rectangle(image_copy, (x, y-30), (x+w, y),
				(0, 0, 255), -1)
			cv2.putText(image_copy, "Non-masked", (x,y-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)		
	
	""" =================== """
	
	cv2.imshow("face detector", image_copy)
	key = cv2.waitKey(1)
	if key == 27:
		break

camera.release()
cv2.destroyAllWindows()