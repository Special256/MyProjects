from keras import backend as K
import tensorflow as tf

model_path = "./my_model.h5"

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

# import model 
saver = tf.keras.models.load_model(model_path)

# Freeze model to pb 
frozen_graph = freeze_session(K.get_session(),"dense_2/Softmax:0")

# Save to ./model/tf_model.pb
# tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)   
tf.io.write_graph(frozen_graph, "model", "tf_model_io.pb", as_text=False)

print("complete!!")


# print(model.summary())

# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_3 (InputLayer)            [(None, 24)]         0                                            
# __________________________________________________________________________________________________
# input_2 (InputLayer)            [(None, 2048)]       0                                            
# __________________________________________________________________________________________________
# embedding (Embedding)           (None, 24, 200)      26000       input_3[0][0]                    
# __________________________________________________________________________________________________
# dropout (Dropout)               (None, 2048)         0           input_2[0][0]                    
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 24, 200)      0           embedding[0][0]                  
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 256)          524544      dropout[0][0]                    
# __________________________________________________________________________________________________
# lstm (LSTM)                     (None, 256)          467968      dropout_1[0][0]                  
# __________________________________________________________________________________________________
# add (Add)                       (None, 256)          0           dense[0][0]                      
#                                                                  lstm[0][0]                       
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 256)          65792       add[0][0]                        
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 130)          33410       dense_1[0][0]                    
# ==================================================================================================
# Total params: 1,117,714
# Trainable params: 1,091,714
# Non-trainable params: 26,000
# __________________________________________________________________________________________________

# print(model.outputs, model.inputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 130) dtype=float32>]
# [<tf.Tensor 'input_2:0' shape=(?, 2048) dtype=float32>, <tf.Tensor 'input_3:0' shape=(?, 24) dtype=float32>]


