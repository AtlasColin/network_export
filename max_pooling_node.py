# -*- coding: utf-8 -*-

from collections import OrderedDict
from max_pooling_get_im2col_table import CMaxPoolingParam, get_input_to_col_table
from proto_py.shape_pb2 import XZY_Shape
from proto_py.node_pb2 import XZY_Node


def read_tf_max_pooling_node(tf_nodes, node_name):  # end with "/MaxPool"
    weight_node_name = "/".join(node_name.split("/")[0:2]) + "/MaxPool"  # max_pooling_1/MaxPool for example
    node_index = list([e.name == weight_node_name for e in tf_nodes]).index(True)
    tf_node = tf_nodes[node_index]

    padding = tf_node.attr["padding"].s
    kernel_height = tf_node.attr["ksize"].list.i[1]
    kernel_width = tf_node.attr["ksize"].list.i[2]
    stride_y = tf_node.attr["strides"].list.i[1]
    stride_x = tf_node.attr["strides"].list.i[2]
    return padding.decode("utf-8"), (kernel_height, kernel_width), (stride_y, stride_x)


def get_col_table_and_output_size(input_shape, kernel_size, stride_size, padding_type):
    assert isinstance(input_shape, XZY_Shape)

    input_channel = input_shape.channel
    input_size = (input_shape.height, input_shape.width)
    param = CMaxPoolingParam(kernel_size=kernel_size,
                             stride_size=stride_size,
                             input_channel=input_channel,
                             padding_type=padding_type)
    input_to_col_table, output_size = get_input_to_col_table(input_size, param)
    return input_to_col_table, output_size


def get_max_pooling_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert isinstance(node_name, str)
    assert isinstance(input_node_names, list) and all([isinstance(e, str) for e in input_node_names])
    assert len(input_node_names) == 1
    assert isinstance(exported_nodes, OrderedDict)

    input_shape = exported_nodes[input_node_names[0]].output_shape
    padding_type, kernel_size, stride_size = read_tf_max_pooling_node(tf_nodes, node_name)
    input_to_col_table, output_size = \
        get_col_table_and_output_size(input_shape, kernel_size, stride_size, padding_type)

    node = XZY_Node()
    node.name = node_name
    node.input_names.extend(input_node_names)
    node.input_shapes.extend([input_shape])
    node.output_shape.channel = input_shape.channel
    node.output_shape.height = output_size[0]
    node.output_shape.width = output_size[1]
    param = node.max_pooling_param
    param.col_matrix_size.height = input_shape.channel * output_size[0] * output_size[1]
    param.col_matrix_size.width = kernel_size[0] * kernel_size[1]
    param.kernel_size.height = kernel_size[0]
    param.kernel_size.width = kernel_size[1]
    param.col_table.extend(input_to_col_table)

    return node


def main():
    pass


if __name__ == "__main__":
    main()
