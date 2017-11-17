# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from separable_convolution_get_im2col_table import CSeparableConvolutionParam, get_input_to_col_table
from proto_py.node_pb2 import XZY_Node


def read_tf_depthwise_weight_node(tf_nodes, node_name):  # end with "/depthwise_weights"
    weight_node_name = "/".join(node_name.split("/")[0:2]) + "/depthwise_weights"  # Network/DepthSepConv_1_depthwise/depthwise_weights for example
    node_index = list([e.name == weight_node_name for e in tf_nodes]).index(True)
    tf_node = tf_nodes[node_index]

    weight_shape = tf_node.attr["value"].tensor.tensor_shape
    weight_raw_data = tf_node.attr["value"].tensor.tensor_content
    # weights的shape为[kernel_height, kernel_width, input_channel, 1]
    input_channel = weight_shape.dim[2].size
    kernel_height = weight_shape.dim[0].size
    kernel_width = weight_shape.dim[1].size
    weight_data = np.frombuffer(weight_raw_data, dtype=np.float32)
    weight_data = np.reshape(weight_data,
                             newshape=[kernel_height, kernel_width, input_channel, 1])
    weight_data = np.transpose(weight_data, axes=[3, 2, 0, 1])
    return weight_data


def read_tf_depthwise_node(tf_nodes, node_name):  # end with "/depthwise"
    weight_node_name = "/".join(node_name.split("/")[0:2]) + "/depthwise"  # Network/DepthSepConv_1_depthwise/depthwise for example
    node_index = list([e.name == weight_node_name for e in tf_nodes]).index(True)
    tf_node = tf_nodes[node_index]

    padding = tf_node.attr["padding"].s
    stride_y = tf_node.attr["strides"].list.i[1]
    stride_x = tf_node.attr["strides"].list.i[2]
    return padding.decode("utf-8"), (stride_y, stride_x)


def read_tf_biases_node(tf_nodes, node_name):  # end with "/biases"
    weight_node_name = "/".join(node_name.split("/")[0:2]) + "/biases"  # Network/DepthSepConv_1_depthwise/biases for example
    try:
        node_index = list([e.name == weight_node_name for e in tf_nodes]).index(True)
        tf_node = tf_nodes[node_index]
        biases_raw_data = tf_node.attr["value"].tensor.tensor_content
        biases_data = np.frombuffer(biases_raw_data, dtype=np.float32)
    except ValueError:
        biases_data = np.ndarray([], dtype=np.float32)
    return biases_data


def get_col_table_and_output_size(input_size, stride_size, padding_type, weight):
    # sepconv中输出通道与输入一致
    output_channel, input_channel, kernel_size = weight.shape[1], weight.shape[1], weight.shape[2:]
    sepconv_param = CSeparableConvolutionParam(kernel_size=kernel_size,
                                               stride_size=stride_size,
                                               input_channel=input_channel,
                                               output_channel=output_channel,
                                               padding_type=padding_type)
    input_to_col_table, output_size = get_input_to_col_table(input_size, sepconv_param)
    return input_to_col_table, output_size


def get_depthwise_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert isinstance(node_name, str)
    assert isinstance(input_node_names, list) and all([isinstance(e, str) for e in input_node_names])
    assert isinstance(exported_nodes, OrderedDict)

    assert len(input_node_names) == 1
    input_shape = exported_nodes[input_node_names[0]].output_shape
    input_size = (input_shape.height, input_shape.width)
    padding_type, stride_size = read_tf_depthwise_node(tf_nodes, node_name)
    weight = read_tf_depthwise_weight_node(tf_nodes, node_name)
    biases = read_tf_biases_node(tf_nodes, node_name)
    if len(biases.shape) == 0:  # there is not biases add operation
        biases = np.zeros(shape=(weight.shape[0],), dtype=np.float32)
    input_to_col_table, output_size = \
        get_col_table_and_output_size(input_size, stride_size, padding_type, weight)

    node = XZY_Node()
    node.name = node_name
    node.input_names.extend(input_node_names)
    node.input_shapes.extend([input_shape])
    # sepconv中输出通道与输入一致
    node.output_shape.channel = weight.shape[1]
    node.output_shape.height = output_size[0]
    node.output_shape.width = output_size[1]
    param = node.separable_convolution_param
    param.kernel_size.height = weight.shape[2]
    param.kernel_size.width = weight.shape[3]
    param.col_matrix_size.height = weight.shape[2] * weight.shape[3]
    param.col_matrix_size.width = output_size[0] * output_size[1]
    param.col_table.extend(input_to_col_table)
    param.weight.extend(weight.flatten())
    param.biases.extend(biases.flatten())

    return node


def main():
    pass


if __name__ == "__main__":
    main()
