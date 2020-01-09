# -*- coding: utf-8 -*-
"""
MASK RCNN

Freezing the graph

Inferencing with the model

COCO Pretrained model

"""

import cv2 , os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import visualize 
from saved_model_preprocess import ForwardModel
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

import coco
# Download h5 model from https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5 
class InferenceConfig(coco.CocoConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  
coco_config = InferenceConfig()
coco_config.display()

# PreProcess Model

preprocess_obj = ForwardModel(coco_config) # config , outputs


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
               'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
               'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
               'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations . Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    
    return ax
  
# build the graph from pb file

def build_graph_and_infer(image , model_path):
  
  images = np.expand_dims(image , axis = 0)
  molded_images , image_metas , windows = preprocess_obj.mold_inputs(images)
  molded_images = molded_images.astype(np.float32)
  image_metas = image_metas.astype(np.float32)
  image_shape = molded_images[0].shape
  
  for g in molded_images[1:]:
    assert g.shape == image_shape, \
    "After resizing , all images must have the same size , Check IMAGE_RESIZE_MODE and image sizes"

  anchors = preprocess_obj.get_anchors(image_shape)
  anchors = np.broadcast_to(anchors , (images.shape[0],)+anchors.shape)
  
  graph_pb = model_path
  
  with tf.gfile.GFile(graph_pb , "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  
  nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Softmax' , 'Placeholder')]
  
  for node in nodes:
    print(node)
  
  
  with tf.Session(graph = tf.Graph() , config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
            
    tf.import_graph_def(graph_def , name = "")
    g = tf.get_default_graph()
    output_dict = {}
    
    for key in ["mrcnn_detection/Reshape_1" , "mrcnn_mask/Reshape_1"]:
      tensor_name = key + ':0'
      output_dict[key] = g.get_tensor_by_name(tensor_name)
    
    input_image = g.get_tensor_by_name("input_image:0")
    
    input_image_meta = g.get_tensor_by_name("input_image_meta:0")
    
    input_anchors = g.get_tensor_by_name("input_anchors:0")
        
    t1 = time.time()
    result = sess.run( output_dict , {input_image : molded_images , input_image_meta : image_metas  , input_anchors: anchors})
    t2 = time.time()
    print("****************      INFERENCE DONE      *************")
    print("  INFERENCE TIME : ",t2-t1)
  result_dict = preprocess_obj.result_to_dict(images , molded_images , windows , result)[0]
  
  print(result_dict)
  ax = get_ax(1)
  
  infer_image = visualize.display_instances(image , result_dict['rois'] , result_dict['mask'] , result_dict['class'] , class_names , result_dict['scores'] , ax = ax , title = "Predictions")
  
  return infer_image

if __name__ == '__main__' :
  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--path_to_image', help ='Path to Image' , required = True)
  parser.add_argument('-m' ,'--path_to_model' , help = 'Path to model' , required = True)
  
  args = vars(parser.parse_args())
  
  image_path = args['path_to_image']
  cwd = os.getcwd()
  image_name = image_path.split('/')[1]
  image_name = image_name.split('.')[0]
  model_path = args['path_to_model']
  
  if not os.path.exists(image_path):
    print("Image_path -- Does not exist")
    exit()
    
  if not os.path.exists(model_path):
    print("Model Path Does not exist")
    exit()
    
  image = cv2.imread(image_path)
  if image is None:
    print("Image path is not proper")
    exit()
    
  infer_image = build_graph_and_infer(image , model_path)
  infer_folder = cwd+"/inferenced_images"
  
  cv2.imwrite(os.path.join(infer_folder , image_name+"_infer.jpg") , infer_image)
  
  
  

      
  
  
  
  
  
