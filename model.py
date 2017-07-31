import csv
import cv2
import numpy as np
import sklearn
import time

####################
# DATA EXPLORATION
####################

# open/read csv log file
y_log = []
train_log = []
test_log = []
with open('data/test.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	next (reader)
	for line in reader:
		test_log.append(line)
	print('Number of lines on test log: ', len(test_log))

with open('data/train.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	next (reader)
	for line in reader:
		y_log.append(float(line[1]))
		train_log.append(line)
	print('Size of y log: ', len(y_log))
	print('Size of train log: ', len(train_log))

# train log prep
for log in train_log:
	del log[1]

def prep_log(t_log):
	for log in t_log:
		if log[10] == 'S':
			log[10] = 1.
		elif log[10] == 'Q':
			log[10] = 2.
		else:
			log[10] = 3.
		if not log[9]:
			log[9] = 1.
		else:
			log[9] = 2.
		if log[7].isdigit():
			log[7] = 1.
		else:
			log[7] = 2.
		if log[3] == "female":
			log[3] = 1.
		else:
			log[3] = 2.
		del log[2]
		del log[0]
		for i in range(len(log)):
			if not log[i]:
				log[i] = 0.
			if isinstance(log[i], str):
				try:
					log[i] = float(log[i])
				except ValueError:
					log[i] = float(int(log[i]))
	return t_log

train_log = prep_log(train_log)
print(train_log[0])

test_log = prep_log(test_log)
print(test_log[0])


# normalize values
a = 0
b = 1.
# pclass
minclass = [min(log) for log in zip(*train_log)]
maxclass = [max(log) for log in zip(*train_log)]
#print('min', minclass, 'max', maxclass)
minval = min(minclass)
#min(val for val in train_log)
maxval = max(maxclass)
#max(val for val in train_log)
#print('min', minval, 'max', maxval)
#for log in train_log:
	#print(log[1])
#norm = np.array(a + ( ( (train_log - minval)*(b - a) )/( maxval - minval ) ) )
#print (norm)

###################
# MODEL TRAINING
###################

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

t = time.time()

train_X_samples, valid_X_samples, train_y_samples, valid_y_samples = train_test_split(train_log, y_log, test_size=0.2)

X_train = np.array(train_X_samples)
y_train = np.array(train_y_samples)
X_train, y_train = shuffle(X_train, y_train)


##########################
# MODEL ARCHITECTURE
##########################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(1, activation='softmax'))

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=20, batch_size=len(X_train))
score = model.evaluate(valid_X_samples, valid_y_samples, batch_size=len(valid_X_samples))
print("Accuracy: ", score)
print("Training duration: ", round(time.time()-t, 2))
print("Model trained")

survival = []
#test_log = np.array(test_log)
#for log in test_log:
survival = model.predict(test_log)
#survival.append([log[0], surv])
print(survival)

"""
#model.add(Convolution2D(12,3,3,input_shape=(12,1,1),subsample=(1,1),activation="relu"))
#model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
#model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Flatten())
#model.add(Dense(100))
#model.add(Dense(50))
#model.add(Dense(10))
#model.add(Dense(1))
#model.compile(optimizer='adam', loss='mse')
#callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0), ModelCheckpoint('model.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only = True, verbose = 0),]

#samples_per_epoch = len(train_samples) // BATCH_SIZE * BATCH_SIZE * 6

#history_object = model.fit_generator(train_generator, samples_per_epoch = samples_per_epoch, validation_data=validation_generator, nb_val_samples=len(validation_samples)//BATCH_SIZE*BATCH_SIZE, nb_epoch=50, verbose=1, callbacks=callbacks)

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.subplot(111)
plt.plot(history_object.history['loss'], 'ro-')
plt.plot(history_object.history['val_loss'], 'bo-')
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show(block=True)
plt.savefig('test.png')
################

# data processing
def process_data(train_log):
	x = train_log

	# normalize image
	#image = (img / 127.5) - 1.0

	return image

# fit_generator function
BATCH_SIZE = 32
def generator(samples, BATCH_SIZE):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, BATCH_SIZE):
			batch_samples = samples[offset:offset+BATCH_SIZE]

			# get images and steering angles
			images = []
			measurements = []
			angle_range = 0.5
			delete_rate = 0.7
			correction = 0.2
			for batch_sample in batch_samples:
				## set filter for steering angle measurement
				measurement = float(batch_sample[3])
				# if the steering angle is not within -0.5 to 0.0 to 0.5
				# include all 3 images in the batch sample
				if (abs(measurement) > angle_range):
					for j in range(3):
						# open/read image files
						source_path = batch_sample[j].split('\\')[-1]
						filename = source_path.split('/')[-1]
						local_path = './data/IMG/' + filename
						image = cv2.imread(local_path)
						image = process_image(image)
						images.append(image)
					measurements.append(measurement)
					measurements.append(measurement + correction)
					measurements.append(measurement - correction)
				else:
					# if not, the delete rate will randomly determine which
					# image/images in the batch sample will be included
					for j in range(3):
						if (np.random.random() > delete_rate):
							# open/read image files
							source_path = batch_sample[j].split('\\')[-1]
							filename = source_path.split('/')[-1]
							local_path = './data/IMG/' + filename
							image = cv2.imread(local_path)
							image = process_image(image)
							images.append(image)
							if (j == 0): measurements.append(measurement)
							if (j == 1): measurements.append(measurement + correction)
							if (j == 2): measurements.append(measurement - correction)

			augmented_images = []
			augmented_measurements = []
			# augment data with flipped version of images and angles
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				flipped_image = cv2.flip(image, 1)
				flipped_measurement = measurement * -1.0
				augmented_images.append(flipped_image)
				augmented_measurements.append(flipped_measurement)

# compile and train the model using the generator function
train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

"""