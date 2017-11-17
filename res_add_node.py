# -*- coding: utf-8 -*-

from proto_py.node_pb2 import XZY_Node


def get_res_add_node(tf_nodes, node_name, input_node_names, exported_nodes):
    assert len(input_node_names) == 2
    input_shape_0 = exported_nodes[input_node_names[0]].output_shape
    input_shape_1 = exported_nodes[input_node_names[1]].output_shape
    assert input_shape_0.height == input_shape_1.height
    assert input_shape_0.width == input_shape_1.width
    node = XZY_Node()
    node.name = node_name
    node.input_names.extend(input_node_names)
    node.input_shapes.extend([input_shape_0])
    node.input_shapes.extend([input_shape_1])
    #  两个输入通道不一定一样的，这里选择与第二输入一致
    node.output_shape.CopyFrom(input_shape_1)

    return node


def main():
    pass


if __name__ == "__main__":
    main()
