# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:43:31 2020

@author: Aleksandra

data last downloaded on 06.09.2020.
930 rows in metadata
180 covid cases x-ray images
45 different findings
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 0.005
EPOCHS = 25
BS = 8

# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights = "imagenet", include_top = False,
                  input_tensor = Input(shape = (64, 64, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output

# pooling
headModel = MaxPooling2D(pool_size = (2, 2))(headModel)

# flattening
headModel = Flatten()(headModel)

# adding a fully connected layer
headModel = Dense(128, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation = "softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs = baseModel.input, outputs = headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False
    
# compile our model
print("[INFO] compiling model...")
optimizer = Adam(learning_rate=0.005, amsgrad=True)
model.compile(optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size = (64, 64),
        batch_size = BS,
        class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size = (64, 64),
        batch_size = BS,
        class_mode = 'categorical')

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
        train_generator,
        steps_per_epoch=36,
        validation_data=validation_generator,
        validation_steps=4,
		epochs=EPOCHS)