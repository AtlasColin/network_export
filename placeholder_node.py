# -*- coding: utf-8 -*-

from collections import OrderedDict
from tensorflow.core.framework.node_def_pb2 import NodeDef
from proto_py.shape_pb2 import XZY_Shape
from proto_py.node_pb2 import XZY_Node


def read_tf_placeholder_node(tf_node):
    assert isinstance(tf_node, NodeDef)

    shape = tf_node.attr["shape"].shape
    height = shape.dim[1].size
    width = shape.dim[2].size
    channel = shape.dim[3].size
    return channel, height, width


def get_placeholder_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert isinstance(node_name, str)
    assert isinstance(input_node_names, list) and all([isinstance(e, str) for e in input_node_names])
    assert isinstance(exported_nodes, OrderedDict)

    index = list([e.name == node_name for e in tf_nodes]).index(True)
    channel, height, width = read_tf_placeholder_node(tf_nodes[index])

    shape = XZY_Shape()
    shape.channel = channel
    shape.height = height
    shape.width = width

    node = XZY_Node()
    node.name = node_name
    node.input_shapes.extend([shape])
    node.output_shape.CopyFrom(shape)

    return node


def main():
    pass


if __name__ == "__main__":
    main()
