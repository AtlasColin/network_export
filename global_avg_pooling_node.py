# -*- coding: utf-8 -*-

from proto_py.node_pb2 import XZY_Node


def get_global_avg_pooling_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert len(input_node_names) == 1
    input_shape = exported_nodes[input_node_names[0]].output_shape

    node = XZY_Node()
    node.name = node_name
    node.input_names.extend(input_node_names)
    node.input_shapes.extend([input_shape])
    node.output_shape.channel = input_shape.channel
    node.output_shape.height = 1
    node.output_shape.width = 1

    return node


def main():
    pass


if __name__ == "__main__":
    main()
