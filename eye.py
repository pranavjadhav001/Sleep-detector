import cv2
import tensorflow as tf
import numpy as np
import glob
import random
import pickle
from sklearn.model_selection import train_test_split
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

MODEL_NAME = 'EYECLOSEDOPEN'
LR = 1e-3
img_size = 24

files = glob.glob(r'F:\notes\dataset_B_Eye_Images\**\*jpg')
closed_data = []
closed_label = []
for file in files:
##    print(file)
    if os.path.basename(file).split('_')[0] == 'closed':
        label = [0,1]
    else:
        label = [1,0]

    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    if img is not None:
        closed_data.append(np.array(img))
        closed_label.append(label)

    else:
        print("image not loaded")

c = list(zip(closed_data,closed_label))
random.shuffle(c)
closed_data , closed_label = zip(*c)


closed_data = np.array(closed_data)
closed_label = np.array(closed_label)

X_train,X_test,y_train,y_test = train_test_split(closed_data,closed_label,test_size=0.2)

X = np.array([i for i in X_train]).reshape(-1,img_size,img_size,1)
Y = [i for i in y_train]

test_x = np.array([i for i in X_test]).reshape(-1,img_size,img_size,1)
test_y = [i for i in y_test]

tf.reset_default_graph()
convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
