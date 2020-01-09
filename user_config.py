import os
# User Define parameters

# Make it True if you want to use the provided coco weights
is_coco = False


ROOT_DIR = os.getcwd()


# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model.h5")
# keras model file path

H5_WEIGHT_PATH = COCO_MODEL_PATH
MODEL_DIR = os.path.dirname(H5_WEIGHT_PATH)

# Path where the Frozen PB will be save
PATH_TO_SAVE_FROZEN_PB = ROOT_DIR

# Name for the Frozen PB name
FROZEN_NAME = 'mask_frozen_graph.pb'


# Version of the serving model
VERSION_NUMBER = 1

# Number of classes that you have trained your model
NUMBER_OF_CLASSES = 80
