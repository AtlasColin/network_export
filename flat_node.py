# -*- coding: utf-8 -*-

from collections import OrderedDict
from proto_py.node_pb2 import XZY_Node


def get_flat_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert isinstance(node_name, str)
    assert isinstance(input_node_names, list) and all([isinstance(e, str) for e in input_node_names])
    assert len(input_node_names) == 1
    assert isinstance(exported_nodes, OrderedDict)

    input_shape = exported_nodes[input_node_names[0]].output_shape

    node = XZY_Node()
    node.name = node_name
    node.input_names.extend(input_node_names)
    node.input_shapes.extend([input_shape])
    node.output_shape.channel = \
        input_shape.channel * input_shape.height * input_shape.width
    node.output_shape.height = 1
    node.output_shape.width = 1

    return node


def main():
    pass


if __name__ == "__main__":
    main()
