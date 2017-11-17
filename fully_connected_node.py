# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from proto_py.node_pb2 import XZY_Node


def read_tf_fully_connected_weight_node(tf_nodes, node_name):  # end with "/weights"
    # Node name like : Network/SE_Module_1/fully_connected/Tensordot
    index = node_name.split("/").index("Tensordot")
    # Network/SE_Module_1/fully_connected/weights for example
    weight_node_name = "/".join(node_name.split("/")[0:index]) + "/weights"
    node_index = list([e.name == weight_node_name for e in tf_nodes]).index(True)
    tf_node = tf_nodes[node_index]

    weight_shape = tf_node.attr["value"].tensor.tensor_shape
    weight_raw_data = tf_node.attr["value"].tensor.tensor_content
    input_channel = weight_shape.dim[0].size
    output_channel = weight_shape.dim[1].size
    weight_data = np.frombuffer(weight_raw_data, dtype=np.float32)
    weight_data = np.reshape(weight_data,
                             newshape=[input_channel, output_channel])
    weight_data = np.transpose(weight_data, axes=[1, 0])
    return weight_data


def read_tf_fully_connected_biases_node(tf_nodes, node_name):  # end with "/biases"
    index = node_name.split("/").index("Tensordot")
    # Network/SE_Module_1/fully_connected/biases for example
    weight_node_name = "/".join(node_name.split("/")[0:index]) + "/biases"
    try:
        node_index = list([e.name == weight_node_name for e in tf_nodes]).index(True)
        tf_node = tf_nodes[node_index]
        biases_raw_data = tf_node.attr["value"].tensor.tensor_content
        biases_data = np.frombuffer(biases_raw_data, dtype=np.float32)
    except ValueError:
        biases_data = np.ndarray([], dtype=np.float32)
    return biases_data


def get_fully_connected_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert isinstance(node_name, str)
    assert isinstance(input_node_names, list) and all([isinstance(e, str) for e in input_node_names])
    assert isinstance(exported_nodes, OrderedDict)

    assert len(input_node_names) == 1
    input_shape = exported_nodes[input_node_names[0]].output_shape
    weight = read_tf_fully_connected_weight_node(tf_nodes, node_name)
    biases = read_tf_fully_connected_biases_node(tf_nodes, node_name)
    if len(biases.shape) == 0:  # there is not biases add operation
        biases = np.zeros(shape=(weight.shape[0],), dtype=np.float32)

    node = XZY_Node()
    node.name = node_name
    node.input_names.extend(input_node_names)
    node.input_shapes.extend([input_shape])
    node.output_shape.channel = weight.shape[0]
    node.output_shape.height = 1
    node.output_shape.width = 1
    param = node.fully_connected_param
    param.weight.extend(weight.flatten())
    param.biases.extend(biases.flatten())

    return node


def main():
    pass


if __name__ == "__main__":
    main()
