
"""
Convert model.ckpt to model.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util

ckpt_path = './checkpoint/data_copyborder_v2/'
#output_node_names = 'CTCBeamSearchDecoder'
output_node_names = 'dense_decode'

# create a session
sess = tf.Session()

# import best model
saver = tf.train.import_meta_graph(ckpt_path+'ocr-model-499001.meta') # graph
saver.restore(sess, ckpt_path+'ocr-model-499001') # variables

# get graph definition
gd = sess.graph.as_graph_def()

# fix batch norm nodes
for node in gd.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in xrange(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']
  elif node.op == 'AssignAdd':
    node.op = 'Add'
    if 'use_locking' in node.attr: del node.attr['use_locking']  

# generate protobuf
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, [output_node_names])
#converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, output_node_names.split(","))
tf.train.write_graph(converted_graph_def, ckpt_path, 'sparse_tensor_to dense_model_v2.pb', as_text=False)
print('complite!')
