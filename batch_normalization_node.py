# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from proto_py.node_pb2 import XZY_Node


def read_tf_moving_mean_node(tf_nodes, node_name):  # end with "/moving_mean"
    # convolution_5/BatchNorm/moving_mean for example
    index = node_name.split("/").index("BatchNorm")
    moving_mean_node_name = "/".join(node_name.split("/")[0:index + 1]) + "/moments/mean"
    node_index = list([e.name == moving_mean_node_name for e in tf_nodes]).index(True)
    tf_node = tf_nodes[node_index]
    moving_mean_raw_data = tf_node.attr["value"].tensor.tensor_content
    moving_mean_data = np.frombuffer(moving_mean_raw_data, dtype=np.float32)
    return moving_mean_data


def read_tf_moving_variance_node(tf_nodes, node_name):  # end with "/moving_variance"
    # convolution_5/BatchNorm/moving_variance for example
    index = node_name.split("/").index("BatchNorm")
    moving_variance_node_name = "/".join(node_name.split("/")[0:index + 1]) + "/moments/variance"
    node_index = list([e.name == moving_variance_node_name for e in tf_nodes]).index(True)
    tf_node = tf_nodes[node_index]
    moving_variance_raw_data = tf_node.attr["value"].tensor.tensor_content
    moving_variance_data = np.frombuffer(moving_variance_raw_data, dtype=np.float32)
    return moving_variance_data


def read_tf_beta_node(tf_nodes, node_name):  # end with "/beta"
    # convolution_5/BatchNorm/beta for example
    index = node_name.split("/").index("BatchNorm")
    beta_node_name = "/".join(node_name.split("/")[0:index + 1]) + "/beta"
    try:
        node_index = list([e.name == beta_node_name for e in tf_nodes]).index(True)
        tf_node = tf_nodes[node_index]
        beta_raw_data = tf_node.attr["value"].tensor.tensor_content
        beta_data = np.frombuffer(beta_raw_data, dtype=np.float32)
    except ValueError:
        beta_data = np.ndarray([], dtype=np.float32)
    return beta_data


def read_tf_gamma_node(tf_nodes, node_name):  # end with "/gamma"
    # convolution_5/BatchNorm/gamma for example
    index = node_name.split("/").index("BatchNorm")
    beta_node_name = "/".join(node_name.split("/")[0:index + 1]) + "/gamma"
    try:
        node_index = list([e.name == beta_node_name for e in tf_nodes]).index(True)
        tf_node = tf_nodes[node_index]
        gamma_raw_data = tf_node.attr["value"].tensor.tensor_content
        gamma_data = np.frombuffer(gamma_raw_data, dtype=np.float32)
    except ValueError:
        gamma_data = np.ndarray([], dtype=np.float32)
    return gamma_data


def get_batch_normalization_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert isinstance(node_name, str)
    assert isinstance(input_node_names, list) and all([isinstance(e, str) for e in input_node_names])
    assert isinstance(exported_nodes, OrderedDict)

    assert len(input_node_names) == 1
    input_shape = exported_nodes[input_node_names[0]].output_shape

    moving_mean_data = read_tf_moving_mean_node(tf_nodes, node_name)
    moving_variance_data = read_tf_moving_variance_node(tf_nodes, node_name)
    gamma_data = read_tf_gamma_node(tf_nodes, node_name)
    if len(gamma_data.shape) == 0:
        gamma_data = np.ones(shape=moving_mean_data.shape, dtype=np.float32)
    beta_data = read_tf_beta_node(tf_nodes, node_name)
    if len(beta_data.shape) == 0:
        beta_data = np.zeros(shape=moving_mean_data.shape, dtype=np.float32)

    node = XZY_Node()
    node.name = node_name
    node.input_names.extend(input_node_names)
    node.input_shapes.extend([input_shape])
    node.output_shape.CopyFrom(input_shape)
    param = node.batch_normalization_param
    param.mean.extend(moving_mean_data.flatten())
    param.variance.extend(moving_variance_data.flatten())
    param.scale.extend(gamma_data.flatten())
    param.offset.extend(beta_data.flatten())

    return node


def main():
    pass


if __name__ == "__main__":
    main()
