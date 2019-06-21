# =========================================================================
# This code built for the Competition of iNaturalist 2019 at FGVC6 
# Part of MCEN90048 project
# This file contains FLAGS definitions
# Written by Zhuo Li
# The Univerivsity of Melbourne
# zhuol7@student.unimelb.edu.au
# =========================================================================

import tensorflow as tf
flags = tf.flags
FLAGS = tf.flags.FLAGS

# Define hyper-parameters here
flags.DEFINE_string('VERSION', 'v_final', 'Version of model.')
flags.DEFINE_string('MODEL', 'Inception-ResNet-v2', 'Pre-train model.')
flags.DEFINE_string('OPTIMIZER', 'Nadam', 'Optimizer.')
flags.DEFINE_float('LR', 1e-4, 'Learning rate.')
flags.DEFINE_boolean('RELOAD', True, 'If reload the save model.')
flags.DEFINE_integer('EPOCH', 30, 'Number of target size.')
flags.DEFINE_integer('TARGET_SIZE', 299, 'Number of target size.')
flags.DEFINE_integer('NUM_CLASSES', 1010, 'Number of class number.')
flags.DEFINE_integer('BATCH_SIZE_TRAIN', 36, 'Batch size of training dataset.')
flags.DEFINE_integer('BATCH_SIZE_VAL', 128, 'Batch size of validation dataset.')
flags.DEFINE_integer('BUFFER_SIZE_TRAIN', 500, 'Buffer size of training dataset.')
flags.DEFINE_integer('BUFFER_SIZE_VAL', 500, 'Buffer size of validation dataset.')
flags.DEFINE_string('IMAGE_FORMAT', 'channels_last', 'The format of images by default.')


# Working directory info
flags.DEFINE_string('SYSPATH', '/home/zhuo/Schrodinger/{}'.format(FLAGS.VERSION), 'Default working folder.')
flags.DEFINE_string('DEFAULT_IN', '/data/cephfs/punim0811/Datasets/iNaturalist/', 'Default input folder.')
flags.DEFINE_string('DEFAULT_OUT', '{}/Results/'.format(FLAGS.SYSPATH), 'Default output folder.')
flags.DEFINE_string('PERDICTED', '{}/submission.csv'.format(FLAGS.SYSPATH), 'The location of predicted.')
flags.DEFINE_string(
    'SAVE_CKPT',
    FLAGS.DEFAULT_OUT+'trial_{}_{}_{}.h5'.format(FLAGS.TARGET_SIZE, FLAGS.VERSION, FLAGS.OPTIMIZER),
    'Default output folder.')
flags.DEFINE_string(
    'DEFAULT_DOWNLOAD', '/home/zhuo/Schrodinger/{}'.format(FLAGS.VERSION),
    'Default folder for downloading large datasets.')
flags.DEFINE_string(
    'INCEPTION_V3',
    '/home/zhuo/Schrodinger/Weight/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'The location of the weight of Inception v3 model.')
flags.DEFINE_string(
    'INCEPTION_RES_V2',
    '/home/zhuo/Schrodinger/Weight/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'The location of the weight of InceptionRestNet v2 model.')



