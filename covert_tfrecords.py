# =========================================================================
# This code built for the Competition of iNaturalist 2019 at FGVC6
# Part of MCEN90048 project
# This file converts the datasets in to tfrecords files
# Written by Zhuo LI
# The Univerivsity of Melbourne
# zhuol7@student.unimelb.edu.au
# =========================================================================

import json
import os.path
from GeneralTools.misc_fun import FLAGS
from GeneralTools.images_to_tfrecords import images_to_tfrecords


key = 'train'  # Choose from {'train', 'val', 'test'}
num_images_per_tfrecord = {'train': 11531, 'val': 3030, 'test': 17675}
num_images_per_tfrecord = num_images_per_tfrecord[key]

# Read json file
annotation_file = '{}2019.json'.format(key)
with open(os.path.join(FLAGS.DEFAULT_DOWNLOAD, annotation_file)) as data_file:
    image_annotations = json.load(data_file)

# Extract image file names and classes if provided
images = image_annotations['images']
annotations = image_annotations['annotations'] if 'annotations' in image_annotations else None
image_names = [image['file_name'] for image in images]
image_class = None if annotations is None else [annotation['category_id'] for annotation in annotations]
image_index = 4

# Debug use
print('The {}-th validation image locates at {}; its class is {}'.format(
    image_index, image_names[image_index], 'unknown' if image_class is None else image_class[image_index]))
print('There are {} images in {}'.format(len(image_names), annotation_file))

# Configure folders to save the data
output_folder = os.path.join(FLAGS.DEFAULT_DOWNLOAD, 'tfrecords_{}/'.format(FLAGS.TARGET_SIZE))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_filename = output_folder + key

# Do the covert
images_to_tfrecords(image_names, output_filename, num_images_per_tfrecord,
                    image_class=image_class, target_size=FLAGS.TARGET_SIZE)

print('The tfrecords are saved to {} successfully'.format(output_filename))