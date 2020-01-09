# -*- coding: utf-8 -*-

"""
Freeze Graph 

"""

import tensorflow as tf
import keras.backend as K
#from tensorflow.python.saved_model import signature_constants
#from tensorflow.python.saved_model import tag_constants
import os
from model import MaskRCNN

cwd = os.getcwd()
H5_MODEL = os.path.join(cwd , "mask_rcnn_coco.h5")
MODEL_DIR = os.path.dirname(H5_MODEL)
PATH_TO_SAVE_FROZEN_PB = cwd+"/model"
NUMBER_OF_CLASSES = 80
FROZEN_NAME = 'frozen_graph.pb'

def get_config():
  import coco
  class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
  config = InferenceConfig()
  return config


def freeze_session(session , keep_var_names = None , output_names = None , clear_devices = True):
  graph = sess.graph
  
  with graph.as_default():
    freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
    
    output_names = output_names or []
    input_graph_def = graph.as_graph_def()
    
    if clear_devices:
      for node in input_graph_def.node:
        node.device = ""
        
    frozen_graph = tf.graph_util.convert_variables_to_constants(session , input_graph_def , output_names  , freeze_var_names)
    
    return frozen_graph
  
def freeze_model(model , name):
  frozen_graph = freeze_session(sess , output_names  = [out.op.name for out in model.outputs][:4])
  directory = PATH_TO_SAVE_FROZEN_PB
  tf.train.write_graph(frozen_graph, directory, name , as_text=False)
  print("Graph Frozen")

config = get_config()
sess = tf.Session()
K.set_session(sess)



model = MaskRCNN(mode = "inference"  , model_dir = MODEL_DIR , config = config)

model.load_weights(H5_MODEL , by_name = True)

freeze_model(model.keras_model , FROZEN_NAME)

print("Freezing model Done")



 



