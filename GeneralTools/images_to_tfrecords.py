# =========================================================================
# This code built for the Competition of iNaturalist 2019 at FGVC6
# Part of MCEN90048 project
# This file contains a founction that converts datasets to tfrecords
# Modified from open sources by Zhuo LI
# The Univerivsity of Melbourne
# zhuol7@student.unimelb.edu.au
# =========================================================================

import sys
import time
import os.path
import tensorflow as tf
from io import BytesIO
from PIL import Image, ImageOps
from GeneralTools.misc_fun import FLAGS

# Define macro
# FloatList, Int64List and BytesList are three base feature types

def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
        if isinstance(value, float) else tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
        if isinstance(value, int) else tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
        if isinstance(value, (str, bytes)) else tf.train.BytesList(value=value))


def images_to_tfrecords(image_names, output_filename, num_images_per_tfrecord, image_class=None, target_size=299):
    """ This function converts images listed in the image_names to tfrecords files

    :param image_names: a list of strings like ['xxx.jpg', 'xxx.jpg', 'xxx.jpg', ...]
    :param output_filename: 'train'
    :param num_images_per_tfrecord: integer
    :param image_class: class label for each image, a list like [1, 34, 228, ...]
    :param target_size: the size of images after padding and resizing
    :return:
    """

    num_images = len(image_names)
    # iteratively handle each image
    writer = None
    start_time = time.time()
    for image_index in range(num_images):
        # retrieve a single image
        im_loc = os.path.join(FLAGS.DEFAULT_DOWNLOAD, image_names[image_index])
        im_cla = image_class[image_index] if isinstance(image_class, list) else None
        im = Image.open(im_loc)

        # resize the image
        old_size = im.size
        ratio = float(target_size) / max(old_size)
        if not ratio == 1.0:
            new_size = tuple([int(x * ratio) for x in old_size])
            im = im.resize(new_size, Image.LANCZOS)

        # zero-pad the images
        new_size = im.size
        delta_w = target_size - new_size[0]
        delta_h = target_size - new_size[1]
        if delta_w < 0 or delta_h < 0:
            raise AttributeError('The target size is smaller than the image size {}.'.format(new_size))
        elif delta_w > 0 or delta_h > 0:
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            im = ImageOps.expand(im, padding)

        # if image not RGB format, convert to RGB
        # This is done in case the image is greyscale
        if im.mode != 'RGB':
            im = im.convert('RGB')

        # convert the full image data to the jpeg compressed string to reduce tfrecord size
        with BytesIO() as fp:
            im.save(fp, format="JPEG")
            im_string = fp.getvalue()

        # save to tfrecords
        if image_index % num_images_per_tfrecord == 0:
            file_out = "{}_{:03d}.tfrecords".format(output_filename, image_index // num_images_per_tfrecord)
            # if file_out exist, raise an error, as that means this job has been done before
            if os.path.isfile(file_out):
                print('Job abortion: {} already exists.'.format(file_out))
                break
            writer = tf.python_io.TFRecordWriter(file_out)
        if image_class is None:
            # for test set, the labels are unknown and not provided
            instance = tf.train.Example(
                features=tf.train.Features(feature={
                    'x': _bytes_feature(im_string)
                }))
        else:
            instance = tf.train.Example(
                features=tf.train.Features(feature={
                    'x': _bytes_feature(im_string),
                    'y': _int64_feature(im_cla)
                }))
        writer.write(instance.SerializeToString())
        if image_index % 2000 == 0:
            sys.stdout.write('\r {}/{} instances finished.'.format(image_index + 1, num_images))
        if image_index % num_images_per_tfrecord == (num_images_per_tfrecord - 1):
            writer.close()

    writer.close()
    duration = time.time() - start_time
    sys.stdout.write('\n All {} instances finished in {:.1f} seconds'.format(num_images, duration))