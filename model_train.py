import codecs
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog 
from sklearn.model_selection import train_test_split
import os
import json

#positive 2165 negative 1930
def read_my_data():
    path = "/home/sarih/presentation/"
    X=[]
    y=[]

    for i in range (1,3000):
        path_img =path+"/positives/"+"n"+str(i)+".jpg"
        if os.path.exists(path_img):
            img = cv2.imread(path_img)   
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #to grey
            img = cv2.resize(img, (128,128))
            features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(4, 4), visualize=False)
            X.append(features)
            y.append(1)
    # for negatives img
        path_img =path+"/negatives/"+"n"+str(i)+".jpg"
        if os.path.exists(path_img):
            img = cv2.imread(path_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128,128))
            features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(4, 4), visualize=False)
            X.append(features)
            y.append(0)
            

    return np.asarray(X), np.asarray(y)

X,y = read_my_data()

# json file
data = np.hstack((X,y.reshape(y.shape[0],1)))
json.dump(data.tolist(),codecs.open("Features_label.json",'w',encoding='utf-8'),
    separators=(',',':'),
    sort_keys=True,
    indent=4)
# X  matrice 2x2


obj_text = codecs.open('Features_label.json', 'r', encoding='utf-8').read()
data     = json.loads(obj_text)
data     = np.array(data)
np.random.shuffle(data)
X        = data[:, 0:(data.shape[1]-3)]
y        = data[:,-1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(64, activation = "relu"),
			tf.keras.layers.Dropout(0.3),
			tf.keras.layers.Dense(2, activation = "sigmoid")
			])

#### create directory that will contain weights
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
													save_weights_only=True,
													verbose=1)										
####

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
				loss = "BinaryCrossentropy",
				metrics=["accuracy"])

model.fit(x_train, y_train, batch_size = 64, epochs=10,
					validation_split=0.2, callbacks=[cp_callback])
print(os.listdir(checkpoint_dir))





print(model.evaluate(x_test, y_test))

tf.keras.models.save_model(model, "MyNeuralNetwork")







# print(X.shape)
# x_train, x_test,y_train, y_test = train_test_split(X, y, test_size=0.3)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation = "relu"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(64, activation = "relu"),
#     tf.keras.layers.Dense(64, activation = "relu"),
#     tf.keras.layers.Dense(2, activation = "softmax"),
# ])

# model.compile(optimizer="adam",
#                 loss="SparseCategoricalCrossentropy",
#                 metrics=["accuracy"])
# model.fit(x_train, y_train, batch_size= 32, epochs=20,validation_split=0.2)
# print(model.evaluate(x_test,y_test))