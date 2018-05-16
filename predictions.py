import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.models import model_from_yaml

import cv2

path0 = '/home/m/salad/deep-salad_data/deep-salad/salad'

# load everything
model = model_from_yaml(open('model2').read())
model.load_weights('weights2')

# the data, shuffled and split between tran and test sets
X_train, Y_train = np.load(path0+'trainsetx.npy'), np.load(path0+'trainsety.npy')
X_test, Y_test = np.load(path0+'testsetx.npy'), np.load(path0+'testsety.npy')
print("X_train original shape", X_train.shape)
print("y_train original shape", Y_train.shape)

#X_train = X_train.reshape(len(X_train), 200, 200, 3)
#X_test = X_test.reshape(len(X_test), 200, 200, 3)
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

for i in range(len(X_test)):
	predictions = model.predict(X_test[i:i+1])
	res = predictions[0][0]
	
	bush = cv2.cvtColor(cv2.resize(X_test[i], (400, 400)), cv2.COLOR_HSV2RGB)
	img = np.zeros([600,400,3])
	cv2.putText(img, 'res:  '+str(res), (10,70),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
	cv2.putText(img, 'true: '+str(Y_test[i]), (10,150),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
	img[200:,:] = bush / 255
	cv2.imshow('img',img)
	cv2.waitKey(2000)
