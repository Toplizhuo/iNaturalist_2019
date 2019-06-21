# =========================================================================
# This code built for the Competition of iNaturalist 2019 at FGVC6 
# Part of MCEN90048 project
# Using the Inception-v3 or Inception-ResNet-v2 model by Keras
# Written by Zhuo LI
# The Univerivsity of Melbourne
# zhuol7@student.unimelb.edu.au
# =========================================================================

# ----------------------------------------------- Configuration -----------------------------------------------

from GeneralTools.misc_fun import FLAGS
from GeneralTools.param_print import param_print
from GeneralTools.inaturalist_func import read_inaturalist

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf

# ------------------------------------------- End of Configuration --------------------------------------------

# --------------------------------------------------- Model ---------------------------------------------------

"""
Read the dataset
All hyper-parameters of training and validation defined in misc_fun.py
Testing hyper-parameters defined as below

"""
dataset_tr, steps_per_tr_epoch = read_inaturalist(
    'train', batch_size=FLAGS.BATCH_SIZE_TRAIN, target_size=FLAGS.TARGET_SIZE, 
    do_augment=True, buffer_size=FLAGS.BUFFER_SIZE_TRAIN)
dataset_va, steps_per_va_epoch = read_inaturalist(
    'val', batch_size=FLAGS.BATCH_SIZE_VAL, target_size=FLAGS.TARGET_SIZE, 
    do_augment=True, buffer_size=FLAGS.BUFFER_SIZE_VAL)
dataset_ts, steps_per_ts_epoch = read_inaturalist(
    'test', batch_size=25, target_size=FLAGS.TARGET_SIZE,
    do_augment=True, buffer_size=500)

"""
Print the hyper-parameters as following format:
==================================================
Vision:            v2.3.4(Sample)
Pre-train model:   Inception-ResNet-v2
==================================================
Batch size:        38(train), 128(validation)
Buffer size:       500(train), 500(validation)
Target size:       299*299
Total epoch:       30
Optimizer:         Nadam
Learning Rate:     0.0001
==================================================
"""
param_print()

"""
Load the pre-train model
If a model have already trained, the previous weight would be loaded
If not, the pre-train model will be loaded
Add customer layers
Configure the model for training

"""
if os.path.isfile(FLAGS.SAVE_CKPT) and FLAGS.RELOAD:
    # load the trained weight
    mdl = load_model(
        FLAGS.SAVE_CKPT,
        custom_objects={'softmax_cross_entropy': tf.losses.softmax_cross_entropy})
    print('Model loaded from {}'.format(FLAGS.SAVE_CKPT))
else:
    # load the pre-train model
    if FLAGS.MODEL == 'Inception-v3':
        base_model = applications.InceptionV3(
            weights=None,
            include_top=False,
            input_shape=(FLAGS.TARGET_SIZE, FLAGS.TARGET_SIZE, 3))
        base_model.load_weights(FLAGS.INCEPTION_V3)
    elif FLAGS.MODEL == 'Inception-ResNet-v2':
        base_model = applications.inception_resnet_v2.InceptionResNetV2(
            weights=None,
            include_top=False,
            input_shape=(FLAGS.TARGET_SIZE, FLAGS.TARGET_SIZE, 3))
        base_model.load_weights(FLAGS.INCEPTION_RES_V2)
    else:
        print('weight loading error')
        sys.exit(1)

    # Add customer layers 
    mdl = Sequential([
        base_model, GlobalAveragePooling2D(FLAGS.IMAGE_FORMAT), Dropout(0.5),
        Dense(FLAGS.NUM_CLASSES, activation='linear')])

    # Configures the model for training
    mdl.compile(
        tf.keras.optimizers.Nadam(lr=FLAGS.LR),
        loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])

mdl.summary()

"""
Train the model

"""
# Do the timing
start_time = time.time()

# Save the model after every epoch
checkpoint = ModelCheckpoint(
    FLAGS.SAVE_CKPT, monitor='val_acc', verbose=1,
    save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Trains the model for a fixed number of epochs
history = mdl.fit(
    dataset_tr.dataset, epochs=FLAGS.EPOCH, callbacks=[checkpoint], validation_data=dataset_va.dataset,
    steps_per_epoch=steps_per_tr_epoch, validation_steps=10, verbose=2)

# Calculate the time
duration = time.time() - start_time
print('\n The training process took {:.1f} seconds'.format(duration))

"""
Predict results with well-trained model on the test dataset
Write down as csv files for the submission

"""
# Do the timing
start_time = time.time()

# Generate output predictions for the input samples
results = mdl.predict(dataset_ts.dataset, verbose=2, steps=steps_per_ts_epoch)
predicted = np.expand_dims(np.argmax(results, axis=1), axis=1)

# Open and read the json files of test dataset to get image id
files = open(FLAGS.DEFAULT_IN + 'test2019.json', 'r')
info = json.load(files)
files.close()
images = info['images']
img_id = np.expand_dims(np.array([image['id'] for image in images]), axis=1)
submission = np.concatenate((img_id, predicted), axis=1)

# Save the predictions to a csv file
print('Predictions saved to {}'.format(FLAGS.PERDICTED))
np.savetxt(FLAGS.PERDICTED, submission, fmt='%d', delimiter=',', header='id,predicted', comments='')

# Calculate the time
duration = time.time() - start_time
print('\n The predicted process took {:.1f} seconds'.format(duration))

# ----------------------------------------------- End of Model ------------------------------------------------
